"""The journal the host holds, and the file it is kept in.

Two things live here, and they are the two halves of one answer. :class:`SqliteTrail` is the backend
that keeps a conversation after the process is gone; :class:`Journal` is the component a host holds,
which owns whichever backend was installed and opens and closes it with the host.

**One file per workspace.** A workspace is the partition — a conversation belongs to the tree it was
held over, not to the program that happened to open it — so the file is named after the workspace's
own path and lives in the user's data directory, never inside the tree: opening the program leaves a
workspace exactly as it found it (:func:`journal_path`). Nothing is created until the journal opens,
so ``--help`` does not make a directory.

**One connection, and a schema that says how far it got.** ``PRAGMA user_version`` plus ordered,
re-runnable steps is how the schema gets to this build's version, and WAL is why a second process can
read what the first is writing. ``":memory:"`` is the other answer: SQLite's own in-memory database, the
same rows and the same migrations and no file at all — a conversation nobody meant to keep.

The file holds three things: the log, a counter per session so turn ids resume where the last process
left them, and a cache of the projections that were read out of the log. The cache is disposable — its
rows are answers that can be computed again, and nothing depends on their being there.

The log itself is a protocol (:mod:`plyngent.kernel.trail`) with more than one implementation, and a
component's field can only be annotated with one contract. This is that contract: the host declares
``trail: Journal``, and ``boot`` decides which backend is installed inside it — the same move as
``Host.model`` being a channel over whichever model was installed rather than the model itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import aiosqlite
import msgspec
from composat import Component
from platformdirs import user_data_path

from plyngent.kernel.model import MODEL_TAGS, AnyMessage
from plyngent.kernel.trail import (
    ATTEMPT_TAG,
    CONSENT_TAG,
    EFFECT_TAG,
    STATE_CHANGE_TAG,
    SUMMARY_TAG,
    USAGE_TAG,
    WITHHELD_TAG,
    AnyRecord,
    Attempt,
    Consent,
    Effect,
    Entry,
    Shape,
    StateChange,
    Summary,
    Trail,
    Usage,
    Withheld,
    as_message,
    kind_of,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The path that asks for a journal which lasts only as long as the process: a conversation nobody meant
#: to keep. Every other path is a file this program opens and migrates.
MEMORY_JOURNAL = ":memory:"

#: How one statement is given its values: in order, or by the names its SQL spells out.
type _Params = Sequence[object] | Mapping[str, object]


def journal_path(root: Path | str) -> Path:
    """Where one workspace's journal lives: a directory of its own in the user's data directory.

    The directory is named for the workspace twice over: a slug from the tree's own name, so a person
    looking at the data directory can tell which is which, and a digest of the resolved path, so two
    workspaces that share a name do not share a history. It is a *pure* answer and creates nothing — the
    directory appears when the journal opens, which is why starting the program with ``--help`` leaves
    the file system exactly as it found it.
    """
    resolved = Path(root).expanduser().resolve()
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:12]
    return user_data_path("plyngent") / "workspaces" / f"{_slug(resolved.name)}-{digest}" / "journal.db"


#: What a directory name may keep of the workspace's own: lowercase words, one dash between them.
_NOT_A_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """A name a directory may carry, from a directory's own — and a fallback for a name with nothing in it."""
    slug = _NOT_A_SLUG.sub("-", name.lower()).strip("-")[:40].rstrip("-")
    return slug or "workspace"


#: Schema version 1: one row per entry, one counter per session for turn ids.
#:
#: ``id`` is the row number and the only thing an entry id is made of (``'e' || id``); ``parent`` is a
#: row of this same table rather than the id's text, because a parent is a row and nothing needs to read
#: one out of the other. Every object is created only if it is absent, which is what makes a step that
#: was interrupted halfway re-runnable: a script that created half its indexes is a script that can run
#: again.
_SCHEMA_1 = """
CREATE TABLE IF NOT EXISTS entry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    turn INTEGER NOT NULL,
    parent INTEGER,
    tag TEXT NOT NULL,
    payload TEXT NOT NULL,
    UNIQUE (session, seq)
);
CREATE INDEX IF NOT EXISTS entry_parent ON entry (session, parent);
CREATE TABLE IF NOT EXISTS turn_no (
    session TEXT PRIMARY KEY,
    next INTEGER NOT NULL
);
"""

#: Schema version 2: what a model was shown for one chain end, so a second read is a row rather than a walk.
#:
#: ``entry`` is a row of ``entry`` and ``projector`` the version of the code that computed this row: the
#: pair is the cache key, and the version is what makes a new projector invalidate everything it made
#: without a migration that has to know what changed.
_SCHEMA_2 = """
CREATE TABLE IF NOT EXISTS projection (
    session TEXT NOT NULL,
    entry INTEGER NOT NULL,
    projector INTEGER NOT NULL,
    messages TEXT NOT NULL,
    PRIMARY KEY (session, entry, projector)
);
"""

#: The schema versions, in order. Step *n* is the answer to "how does the log look at version *n*", and
#: the list is append-only: a change to a released step is a new step.
MIGRATIONS: tuple[str, ...] = (_SCHEMA_1, _SCHEMA_2)

#: Which projection this build computes. It is part of every cache key, so raising it makes every stored
#: projection unreachable at once — which is the whole invalidation story, because the cache is
#: rebuildable and may therefore be dropped rather than migrated. Version 2 reads a ``summary`` as the
#: message it stands for; version 1 skipped it, and nothing it wrote can be true of a chain that has one.
#: Version 3 honours a prune (:class:`~plyngent.kernel.trail.Withheld`): a chain that carries one no
#: longer projects to what version 2 wrote down for it.
PROJECTOR = 3

_COLUMNS = "SELECT id, seq, turn, parent, tag, payload FROM entry"

#: The tags a conversation continues from: what a model reads, plus the records that sit *in* a chain
#: without being messages — the summary that stands in for what a model read before, the prune that says
#: what of it is not shown, and the effect that says what one call left behind. One set, two uses — the
#: placeholder list for the one query that names them (:meth:`head`) and the filter a projection applies
#: — so what a turn may continue and what a model is shown cannot drift.
#:
#: A prune belongs here and not among the model tags: nothing reads a prune *as a message*, and it must
#: still be the place a turn grows from — a prune that did not become the head would hide nothing from
#: the next turn, which is the same bug a summary that did not become the head has (E53). An effect is
#: the same shape for a different reason: it is written below the result it belongs to, so a turn that
#: stopped right after it would otherwise grow from the result *above* it and leave the effect on a
#: branch of its own. Projecting from one is unchanged — an effect is not a message
#: (:func:`~plyngent.kernel.trail.as_message`) — so continuing from it says the same thing.
_CONVERSATION_TAGS: frozenset[str] = frozenset(MODEL_TAGS) | {SUMMARY_TAG, WITHHELD_TAG, EFFECT_TAG}
_CONVERSATION = tuple(sorted(_CONVERSATION_TAGS))
_CONVERSATION_IN = ", ".join("?" for _ in _CONVERSATION)

#: One append: the number, the parent check and the row, as one statement.
#:
#: The number comes from this session's own log, and the parent check is the ``WHERE``: a parent that
#: does not exist inserts nothing, and an insert that wrote nothing is the refusal. Doing it this way
#: means the whole append is one transaction SQLite takes the write lock for, so two writers cannot read
#: the same number and one of them lose.
_APPEND = """
INSERT INTO entry (session, seq, turn, parent, tag, payload)
SELECT :session,
       COALESCE((SELECT MAX(seq) FROM entry WHERE session = :session), 0) + 1,
       :turn, :parent, :tag, :payload
 WHERE :parent IS NULL
    OR EXISTS (SELECT 1 FROM entry WHERE session = :session AND id = :parent)
RETURNING id, seq
"""

#: One entry's chain, from the entry named down to its root.
#:
#: ``depth`` counts the hops from the entry asked about, so ``ORDER BY depth DESC`` puts the root first —
#: the order a projection is read in. The recursion walks ``parent``, a row of the same table, so a chain
#: is a join rather than a look-up per entry. A cycle is impossible by construction: a parent must
#: already exist when the row that names it is appended, so every parent has a smaller row number and the
#: walk strictly descends.
_CHAIN = """
WITH RECURSIVE chain(id, parent, tag, payload, depth) AS (
    SELECT id, parent, tag, payload, 0 FROM entry WHERE session = ? AND id = ?
    UNION ALL
    SELECT e.id, e.parent, e.tag, e.payload, c.depth + 1
      FROM entry e JOIN chain c ON e.id = c.parent
)
SELECT id, tag, payload FROM chain ORDER BY depth DESC
"""


class SqliteTrail(Trail):
    """The journal in one SQLite file: the log, the turn counters, and the projections it has answered.

    One connection, opened when the journal comes up and given back when it goes down, and one lock
    around the calls that use it — not because SQLite could not take two readers, but because every
    answer here is a step the rest of the kernel is waiting on, and one at a time is what a single
    connection can honestly promise. WAL and a busy timeout are what let a second process read the same
    file without the first one's write turning into an error.

    **An id is opaque.** Nothing outside this class reads one, and what this class reads out of one is
    its own business: the row an id was minted from. An id that names no row of this shape is a name for
    nothing, and is answered as such rather than as a parse error.
    """

    _path: Path
    _db: aiosqlite.Connection | None
    _lock: asyncio.Lock

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._db = None
        self._lock = asyncio.Lock()

    # -- the file ------------------------------------------------------------------

    @override
    async def open(self) -> None:
        """Make the directory, open the file, and bring the schema to this build's version.

        The directory is made off the loop because it is a file system call, and nothing in this kernel
        blocks the loop: a workspace on a slow mount would otherwise stop every child on a terminal for
        as long as the `mkdir` took.
        """
        if self._db is not None:
            return
        if not self._temporary:
            await asyncio.to_thread(self._path.parent.mkdir, parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self._path), isolation_level=None)
        self._db = db
        try:
            # WAL: a reader does not wait for a writer, and a crash leaves a log another process can
            # recover rather than one that has to be written again from the start.
            _ = await db.execute("PRAGMA journal_mode = WAL")
            _ = await db.execute("PRAGMA busy_timeout = 5000")
            await self._migrate(db)
        except BaseException:
            self._db = None
            await db.close()
            raise

    @override
    async def close(self) -> None:
        """Give the file back; a journal already closed answers this by doing nothing."""
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    @property
    def _temporary(self) -> bool:
        """Whether this journal has no file at all, and therefore nothing to make before opening it."""
        return str(self._path) == MEMORY_JOURNAL

    async def _migrate(self, db: aiosqlite.Connection) -> None:
        """Bring the file up to :data:`MIGRATIONS`, one step at a time, and record how far it got.

        The version is the answer to "which steps have been applied", so a step is applied and then
        counted: a process that dies between the two re-runs the step, which is why every statement is
        written to be harmless the second time.
        """
        version = await self._version(db)
        for step, script in enumerate(MIGRATIONS[version:], start=version + 1):
            _ = await db.executescript(script)
            # Not a parameter: SQLite does not accept one in a PRAGMA, and this is an integer this
            # module owns, not anything a caller said.
            _ = await db.executescript(f"PRAGMA user_version = {step}")

    @staticmethod
    async def _version(db: aiosqlite.Connection) -> int:
        """How many migration steps this file has already been through."""
        async with db.execute("PRAGMA user_version") as cursor:
            row = await cursor.fetchone()
        return 0 if row is None else int(row[0])

    # -- the log -------------------------------------------------------------------

    @override
    async def append(
        self,
        session: str,
        payload: AnyRecord,
        *,
        begin_after: str | None = None,
        turn: int = 0,
    ) -> Entry:
        """Number the entry, record it, and answer the row — refusing a parent this session lacks.

        One statement, because the three things it does are one fact: the number is read from the log, the
        parent is checked against it, and the row is written — all inside SQLite's own write lock, so no
        reader ever sees an entry whose number was taken or whose parent is missing. The parent check is a
        ``WHERE`` clause rather than a separate query: a parent that does not exist inserts nothing, and an
        insert that wrote nothing is the refusal.
        """
        if begin_after is None:
            parent = None
        else:
            parent = _row_of(begin_after)
            if parent is None:
                raise ValueError(_no_such_entry(begin_after, session))
        row = await self._execute(
            _APPEND,
            {
                "session": session,
                "turn": turn,
                "parent": parent,
                "tag": _tag_of(payload),
                "payload": _encode(payload),
            },
        )
        if row is None:
            # An insert with no parent always writes a row, so a write that did nothing can only be a
            # parent this session does not have.
            assert begin_after is not None
            raise ValueError(_no_such_entry(begin_after, session))
        return Entry(
            id=f"e{row[0]}",
            session=session,
            seq=int(row[1]),
            turn=turn,
            begin_after=begin_after,
            payload=payload,
        )

    @override
    async def entry(self, session: str, entry_id: str) -> Entry | None:
        """One row, or ``None`` when this session has no such id."""
        row = _row_of(entry_id)
        if row is None:
            return None
        found = await self._read(_COLUMNS + " WHERE session = ? AND id = ?", (session, row))
        return None if found is None else _entry(session, found)

    @override
    async def head(self, session: str) -> Entry | None:
        """The newest entry a conversation continues from — the bookkeeping does not count."""
        found = await self._read(
            _COLUMNS + f" WHERE session = ? AND tag IN ({_CONVERSATION_IN}) ORDER BY seq DESC LIMIT 1",
            (session, *_CONVERSATION),
        )
        return None if found is None else _entry(session, found)

    @override
    async def derive(self, session: str, begin_after: str | None) -> tuple[AnyMessage, ...]:
        """Walk from ``begin_after`` to its root and answer the tags a model understands.

        The walk is the database's own recursion: one query returns the chain and the order it is read
        in, because a parent is a row and a chain is a join. Rows carrying a tag no model reads are
        dropped here rather than refused, so a file written by a build that keeps more than a
        conversation still projects to one. A prune is the one bookkeeping record that is read before it
        is dropped: what it names is not handed over, and everything else is — which is why the
        projector's version moved when it arrived.

        What came out is kept under ``(session, entry, projector)``, because a projection is a function
        of the chain and a chain never changes: nothing rewrites a row, so a cached answer stays true
        until the projector itself is replaced (§R4 6). The cache is read first and written after, and
        the write may be lost without changing an answer — all a lost write costs is the walk again.
        """
        if begin_after is None:
            return ()
        row = _row_of(begin_after)
        if row is None:
            raise ValueError(_broken_chain(begin_after, session))
        cached = await self._read(
            "SELECT messages FROM projection WHERE session = ? AND entry = ? AND projector = ?",
            (session, row, PROJECTOR),
        )
        if cached is not None:
            return tuple(msgspec.json.decode(str(cached[0]), type=list[AnyMessage]))
        chain = await self._read_all(_CHAIN, (session, row))
        if not chain:
            raise ValueError(_broken_chain(begin_after, session))
        records = [(f"e{entry}", _decode(tag, payload)) for entry, tag, payload in chain if tag in _CONVERSATION_TAGS]
        # What a prune takes away is decided in one pass before anything is read out of the chain: a
        # prune is a row like any other and its targets are older than it, but the entry it hides may sit
        # anywhere below, so the answer has to be known before the first message is handed over.
        withheld = {target for _entry, record in records if isinstance(record, Withheld) for target in record.targets}
        messages = tuple(
            message
            for entry, record in records
            if entry not in withheld and (message := as_message(record)) is not None
        )
        remembered = msgspec.json.encode(list(messages)).decode()
        _ = await self._execute(
            "INSERT OR REPLACE INTO projection (session, entry, projector, messages) VALUES (?, ?, ?, ?)",
            (session, row, PROJECTOR, remembered),
        )
        return messages

    @override
    async def state_of(self, session: str) -> Mapping[str, str]:
        """Everything this session has written down about itself, folded last-write-wins per key.

        Read as the session's own column rather than along a chain, because a fact about a session is
        not a fact about a branch of it: a conversation that disclosed a group has disclosed it, and a
        branch that grew from before the disclosure is still part of that conversation.

        The fold is a dict built in arrival order, so a later entry overwrites an earlier one and no
        reader has to know that a key was written twice. What a setting *means* is the writer's
        business: this answers text, and a caller that wants a value parses it.
        """
        rows = await self._read_all(
            "SELECT payload FROM entry WHERE session = ? AND tag = ? ORDER BY seq",
            (session, STATE_CHANGE_TAG),
        )
        folded: dict[str, str] = {}
        for (payload,) in rows:
            record = msgspec.json.decode(str(payload), type=StateChange)
            folded[record.key] = record.value
        return folded

    @override
    async def shape(self, session: str) -> Shape:
        """Heads and roots, both in arrival order so two reads of an unchanged log agree."""
        rows = await self._read_all("SELECT id, parent FROM entry WHERE session = ? ORDER BY seq", (session,))
        continued = {parent for _, parent in rows if parent is not None}
        return Shape(
            heads=tuple(f"e{row}" for row, _parent in rows if row not in continued),
            roots=tuple(f"e{row}" for row, parent in rows if parent is None),
        )

    @override
    async def entries(self, session: str) -> tuple[Entry, ...]:
        """Every row this session has, in arrival order — one query, one decode per row.

        Arrival order is ``seq`` and not the row number, because a row number is this file's business:
        two journals that grew in the same order number their rows the same way today and nothing
        promises they will tomorrow. Nothing is filtered here, so a caller sees the bookkeeping a
        projection drops, which is the whole difference between "what happened" and "what was said".
        """
        rows = await self._read_all(_COLUMNS + " WHERE session = ? ORDER BY seq", (session,))
        return tuple(_entry(session, row) for row in rows)

    @override
    async def sessions(self) -> tuple[str, ...]:
        """Every conversation this file holds, in the order the first entry of each arrived.

        ``MIN(id)`` rather than ``MIN(seq)``: a sequence number is per session — every session's first
        entry is number one — so only the row number is comparable across sessions.
        """
        rows = await self._read_all("SELECT session FROM entry GROUP BY session ORDER BY MIN(id)", ())
        return tuple(str(session) for (session,) in rows)

    @override
    async def next_turn(self, session: str) -> int:
        """The next turn id for this session, allocated here so a stored log resumes it (D17).

        One statement, and the counter is the row: inserting the first id and bumping a later one are
        the same write, so two processes cannot hand out the same number.
        """
        row = await self._execute(
            "INSERT INTO turn_no (session, next) VALUES (?, 1) "
            "ON CONFLICT (session) DO UPDATE SET next = next + 1 RETURNING next",
            (session,),
        )
        return 1 if row is None else int(row[0])

    # -- the connection ------------------------------------------------------------

    def _connection(self) -> aiosqlite.Connection:
        """The open connection, or a refusal: a journal used before it is up is a bug, not a state."""
        db = self._db
        if db is None:
            msg = f"the journal at {self._path} is not open"
            raise RuntimeError(msg)
        return db

    async def _execute(self, sql: str, params: _Params) -> tuple[Any, ...] | None:
        """Run one statement under the lock, and answer the row it returned if it returned one."""
        async with self._lock, self._connection().execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return None if row is None else tuple(row)

    async def _one(self, db: aiosqlite.Connection, sql: str, params: _Params) -> tuple[Any, ...] | None:
        """One row of a query on a connection the caller already holds, or ``None``."""
        rows = await self._all(db, sql, params)
        return rows[0] if rows else None

    async def _read(self, sql: str, params: _Params) -> tuple[Any, ...] | None:
        """One row of a query, under the lock that keeps this connection's statements in order."""
        async with self._lock:
            return await self._one(self._connection(), sql, params)

    async def _read_all(self, sql: str, params: _Params) -> list[tuple[Any, ...]]:
        """Every row of a query, under the same lock."""
        async with self._lock:
            return await self._all(self._connection(), sql, params)

    @staticmethod
    async def _all(db: aiosqlite.Connection, sql: str, params: _Params) -> list[tuple[Any, ...]]:
        """Every row of a query, as plain tuples — the shapes this module reads and nothing more."""
        async with db.execute(sql, params) as cursor:
            return [tuple(row) for row in await cursor.fetchall()]


class Journal(Component, Trail):
    """One backend, and the six calls that reach it.

    Nothing here adds to what a trail means: every answer is the backend's own, handed through
    unchanged. What it adds is a lifecycle — one place that knows when the backend is open — and one
    contract for a field whose implementation is chosen at install time.
    """

    _backend: Trail

    def __init__(self, backend: Trail) -> None:
        super().__init__()
        self._backend = backend

    @override
    async def load(self) -> None:
        """Open the backend when the host comes up, so the first turn finds it ready."""
        await self.open()

    @override
    async def unload(self) -> None:
        """Close it — the host's last move, after every turn has been settled."""
        await self.close()

    @override
    async def open(self) -> None:
        """Open the backend, which is what :meth:`load` is for; calling it twice is the backend's answer."""
        await self._backend.open()

    @override
    async def close(self) -> None:
        """Close the backend; the path a host takes down and the one a caller closes by hand agree."""
        await self._backend.close()

    @override
    async def append(
        self,
        session: str,
        payload: AnyRecord,
        *,
        begin_after: str | None = None,
        turn: int = 0,
    ) -> Entry:
        """Put one entry at the end of ``begin_after``'s chain, or start a root."""
        return await self._backend.append(session, payload, begin_after=begin_after, turn=turn)

    @override
    async def entry(self, session: str, entry_id: str) -> Entry | None:
        """One row, or ``None`` when this session has no such id."""
        return await self._backend.entry(session, entry_id)

    @override
    async def head(self, session: str) -> Entry | None:
        """The newest entry a conversation continues from, which is where a turn may grow."""
        return await self._backend.head(session)

    @override
    async def derive(self, session: str, begin_after: str | None) -> tuple[AnyMessage, ...]:
        """The conversation a model is given, as the projection from ``begin_after`` to its root."""
        return await self._backend.derive(session, begin_after)

    @override
    async def state_of(self, session: str) -> Mapping[str, str]:
        """What this session knows about itself, folded from what its backend holds."""
        return await self._backend.state_of(session)

    @override
    async def shape(self, session: str) -> Shape:
        """This session's heads and roots, in arrival order."""
        return await self._backend.shape(session)

    @override
    async def entries(self, session: str) -> tuple[Entry, ...]:
        """Every row this session has, in arrival order, as the backend holds it."""
        return await self._backend.entries(session)

    @override
    async def sessions(self) -> tuple[str, ...]:
        """Every conversation this journal holds, in the order each one's first entry arrived."""
        return await self._backend.sessions()

    @override
    async def next_turn(self, session: str) -> int:
        """The next turn id for this session, allocated by the backend so a stored log can resume it."""
        return await self._backend.next_turn(session)


def _row_of(entry_id: str) -> int | None:
    """The row an entry id was minted from, or ``None`` when it names nothing this journal made."""
    prefix, digits = entry_id[:1], entry_id[1:]
    if prefix != "e" or not (digits.isascii() and digits.isdigit()):
        return None
    return int(digits)


def _tag_of(payload: AnyRecord) -> str:
    """What kind of record this payload is, kept in its own column so a projection can skip it unread."""
    return kind_of(payload)


def _encode(payload: AnyRecord) -> str:
    """One record as the text a row keeps; the tag is stored beside it and is not lost by encoding."""
    return msgspec.json.encode(payload).decode()


#: The type each tag is read back as: one row per kind of record, which is what a new kind of row adds.
#:
#: A table rather than a branch, because the tag is already a column: ``msgspec`` refuses a union whose
#: members disagree about the name of their tag field, and these do — a message is tagged ``role`` while
#: every other kind is tagged ``type`` — so reading one row is a lookup and no JSON guessing. A tag
#: nothing here knows is read as a message, which is the only other thing a row could be.
_RECORDS: Mapping[str, type[AnyRecord]] = {
    STATE_CHANGE_TAG: StateChange,
    SUMMARY_TAG: Summary,
    ATTEMPT_TAG: Attempt,
    WITHHELD_TAG: Withheld,
    EFFECT_TAG: Effect,
    CONSENT_TAG: Consent,
    USAGE_TAG: Usage,
}


def _decode(tag: str, payload: str) -> AnyRecord:
    """One record back out of its row, by the tag the row already keeps for exactly this reason."""
    known = _RECORDS.get(tag)
    if known is None:
        return _message(payload)
    return msgspec.json.decode(payload, type=known)


def _message(payload: str) -> AnyMessage:
    """One message back out of its row, for a projection or for a record that is not a state change."""
    return msgspec.json.decode(payload, type=AnyMessage)


def _entry(session: str, row: tuple[Any, ...]) -> Entry:
    """One row as the value everything outside reads it as."""
    row_id, seq, turn, parent, tag, payload = row
    return Entry(
        id=f"e{row_id}",
        session=session,
        seq=int(seq),
        turn=int(turn),
        begin_after=None if parent is None else f"e{parent}",
        payload=_decode(str(tag), str(payload)),
    )


def _no_such_entry(entry_id: str, session: str) -> str:
    """Why an append named a parent this session does not have."""
    return f"no entry {entry_id!r} in session {session}: an entry can only continue one that exists"


def _broken_chain(entry_id: str, session: str) -> str:
    """Why a projection named an entry this session does not have."""
    return f"no entry {entry_id!r} in session {session}: the chain from {entry_id!r} is broken"
