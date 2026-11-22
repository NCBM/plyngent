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

The file holds four things: the log, a counter per session so turn ids resume where the last process
left them, a cache of the projections that were read out of the log, and the snapshots a fold of that
log is kept in. Both of the last two are disposable — their rows are answers that can be computed
again, and nothing depends on their being there.

The log itself is a protocol (:mod:`plyngent.kernel.trail`) with more than one implementation, and a
component's field can only be annotated with one contract. This is that contract: the host declares
``trail: Journal``, and ``boot`` decides which backend is installed inside it — the same move as
``Host.model`` being a channel over whichever model was installed.
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
    DELETED_TAG,
    EFFECT_TAG,
    NOTICE_TAG,
    PROMOTED_TAG,
    STATE_CHANGE_TAG,
    SUMMARY_TAG,
    TURN_ENDED_TAG,
    USAGE_TAG,
    WITHHELD_TAG,
    AnyRecord,
    Attempt,
    Consent,
    Deleted,
    Effect,
    Entry,
    Notice,
    Promoted,
    Shape,
    Snapshot,
    StateChange,
    Summary,
    Trail,
    TurnEnded,
    Usage,
    Withheld,
    as_message,
    kind_of,
)
from plyngent.wire.event import TurnKind

if TYPE_CHECKING:
    from collections.abc import Sequence

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


#: Schema version 1: one row per entry, one counter per session for turn ids, one cached projection.
#:
#: ``id`` is the row number and the only thing an entry id is made of (``'e' || id``); ``parent`` is a
#: row of this same table, because a parent is a row and nothing needs to read
#: one out of the other. Every object is created only if it is absent, which is what makes a step that
#: was interrupted halfway re-runnable: a script that created half its indexes is a script that can run
#: again.
#:
#: ``projection`` is what a model was shown for one chain end, so a second read is a row:
#: walk: ``entry`` is a row of ``entry`` and ``projector`` the code that computed it, and the pair is the
#: cache key.
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
CREATE TABLE IF NOT EXISTS projection (
    session TEXT NOT NULL,
    entry INTEGER NOT NULL,
    projector INTEGER NOT NULL,
    messages TEXT NOT NULL,
    PRIMARY KEY (session, entry, projector)
);
"""

#: Schema version 2: one table for a session's snapshots, beside the log they are derived from.
#:
#: A snapshot is not an entry and is not in the chain: nothing continues from it, it moves no id and no
#: cursor, and dropping the whole table changes no answer (E177). The primary key is ``(session, seq)``
#: because a fold that reached a row is one fold — a second snapshot of the same row replaces it.
_SCHEMA_2 = """
CREATE TABLE IF NOT EXISTS checkpoint (
    session TEXT NOT NULL,
    seq INTEGER NOT NULL,
    version INTEGER NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session, seq)
);
"""

#: The schema versions, in order. Step *n* is the answer to "how does the log look at version *n*", and
#: the list is append-only: a change to a released step is a new step.
MIGRATIONS: tuple[str, ...] = (_SCHEMA_1, _SCHEMA_2)

#: Which snapshot shape this build writes. It is stored in every checkpoint row, so raising it makes
#: every stored snapshot unreachable at once — the same invalidation story as :data:`PROJECTOR`, and the
#: reason a snapshot is safe to keep in a file a later build opens (E177).
CHECKPOINTS = 1

#: How many snapshots of one session are kept. Nothing reads more than the newest; the rest are there
#: for a person comparing what a fold reached, and one is the number a bug would be found with (E177).
KEEP_CHECKPOINTS = 3

#: How many rows one conversation may gather before its fold is kept, when the program was not told.
#: A number rather than "every row", because the write is not free and the fold it saves is a walk that
#: has to happen anyway; ``[journal] checkpoint_rows`` is where a program says otherwise (E177).
DEFAULT_CHECKPOINT_ROWS = 64

#: Which projection this build computes. It is part of every cache key, so raising it makes every stored
#: projection unreachable at once — which is the whole invalidation story, because the cache is
#: rebuildable and may therefore be dropped and rebuilt. It moved to 2 when an aside's rows stopped
#: being part of the conversation a model is given (E163).
PROJECTOR = 2

_COLUMNS = "SELECT id, seq, turn, parent, tag, payload FROM entry"

#: The tags a conversation continues from: what a model reads, plus the records that sit *in* a chain
#: without being messages — the summary that stands in for what a model read before, the prune that says
#: what of it is not shown, and the effect that says what one call left behind. One set, two uses — the
#: placeholder list for the one query that names them (:meth:`head`) and the filter a projection applies
#: — so what a turn may continue and what a model is shown cannot drift.
#:
#: A notice is a message of its own (``note``) but its row is filed under ``notice``, which is why it is
#: named here and not inside ``MODEL_TAGS``: the tags there are the ones a row *is*, and this
#: one is a record that a projection turns into a message.
#:
#: A prune belongs here and not among the model tags: nothing reads a prune *as a message*, and it must
#: still be the place a turn grows from — a prune that did not become the head would hide nothing from
#: the next turn, which is the same bug a summary that did not become the head has (E53). An effect is
#: the same shape for a different reason: it is written below the result it belongs to, so a turn that
#: stopped right after it would otherwise grow from the result *above* it and leave the effect on a
#: branch of its own. Projecting from one is unchanged — an effect is not a message
#: (:func:`~plyngent.kernel.trail.as_message`) — so continuing from it says the same thing.
_CONVERSATION_TAGS: frozenset[str] = frozenset(MODEL_TAGS) | {
    SUMMARY_TAG,
    NOTICE_TAG,
    WITHHELD_TAG,
    EFFECT_TAG,
}
_CONVERSATION = tuple(sorted(_CONVERSATION_TAGS))
_CONVERSATION_IN = ", ".join("?" for _ in _CONVERSATION)

#: The tags a projection reads without handing over: the row that says which kind of turn a stretch
#: belongs to, and the row that says a person kept it. Neither is a message, and neither is a
#: conversation tag — a turn does not grow from them, so they are read here and not in ``head`` (E163).
_KEEPING_TAGS: frozenset[str] = frozenset({TURN_ENDED_TAG, PROMOTED_TAG})

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
#: is a join, with no look-up per entry. A cycle is impossible by construction: a parent must
#: already exist when the row that names it is appended, so every parent has a smaller row number and the
#: walk strictly descends.
_CHAIN = """
WITH RECURSIVE chain(id, parent, tag, payload, turn, depth) AS (
    SELECT id, parent, tag, payload, turn, 0 FROM entry WHERE session = ? AND id = ?
    UNION ALL
    SELECT e.id, e.parent, e.tag, e.payload, e.turn, c.depth + 1
      FROM entry e JOIN chain c ON e.id = c.parent
)
SELECT id, tag, payload, turn FROM chain ORDER BY depth DESC
"""

#: One snapshot, written in place of one that reached the same row.
_PUT_CHECKPOINT = "INSERT OR REPLACE INTO checkpoint (session, seq, version, payload) VALUES (?, ?, ?, ?)"

#: What one session keeps of them: the newest few by the row they reached, and the older ones dropped.
#: A snapshot is derived, so dropping one loses a shortcut and nothing else (E177).
_TRIM_CHECKPOINTS = f"""
DELETE FROM checkpoint
 WHERE session = ?
   AND seq NOT IN (SELECT seq FROM checkpoint WHERE session = ? ORDER BY seq DESC LIMIT {KEEP_CHECKPOINTS})
"""


class SqliteTrail(Trail):
    """The journal in one SQLite file: the log, the turn counters, and the projections it has answered.

    One connection, opened when the journal comes up and given back when it goes down, and one lock
    around the calls that use it: every answer here is a step the rest of the kernel is waiting on, and
    one at a time is what a single connection can honestly promise — not because SQLite could never take
    two readers. WAL and a busy timeout are what let a second process read the same
    file without the first one's write turning into an error.

    **An id is opaque.** Nothing outside this class reads one, and what this class reads out of one is
    its own business: the row an id was minted from. An id that names no row of this shape is a name for
    nothing, and is answered as a name for nothing, which is not a parse error.
    """

    _path: Path
    _db: aiosqlite.Connection | None
    _lock: asyncio.Lock
    _checkpoint_rows: int
    _since: dict[str, int]

    def __init__(self, path: Path | str, *, checkpoint_rows: int | None = None) -> None:
        self._path = Path(path)
        self._db = None
        self._lock = asyncio.Lock()
        #: How many rows one conversation may gather before its fold is kept, and how many each has
        #: gathered since the last one. The count is this process's — a reopened conversation folds once
        #: when it is opened and then counts from there (E177).
        self._checkpoint_rows = DEFAULT_CHECKPOINT_ROWS if checkpoint_rows is None else checkpoint_rows
        self._since = {}

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
            # recover from, with no rewriting from the start.
            _ = await db.execute("PRAGMA journal_mode = WAL")
            _ = await db.execute("PRAGMA busy_timeout = 5000")
            await self._migrate(db)
        except BaseException:
            self._db = None
            await db.close()
            raise

    @override
    async def close(self) -> None:
        """Keep the fold of every conversation this process wrote to, then give the file back.

        Closing is the last boundary where the log is still open, so it is where a conversation that
        nobody finished is folded: the rows since its last snapshot are the ones this process appended,
        which is the cheapest fold there is (E177). A journal already closed answers this by doing
        nothing, and a fold that fails still closes the file.
        """
        db = self._db
        if db is None:
            return
        try:
            for session in sorted(self._since):
                _ = await self.snapshot(session)
        finally:
            self._db = None
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
        ``WHERE`` clause: a parent that does not exist inserts nothing, and an insert that wrote nothing is
        the refusal.
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
        self._since[session] = self._since.get(session, 0) + 1
        if self._since[session] >= self._checkpoint_rows or isinstance(payload, Withheld):
            _ = await self.snapshot(session)
            self._since[session] = 0
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
        dropped here, so a file written by a build that keeps more than a conversation still projects to
        one. A prune is the one bookkeeping record that is read before it
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
        records = [
            (f"e{entry}", int(turn), _decode(tag, payload))
            for entry, tag, payload, turn in chain
            if tag in _CONVERSATION_TAGS or tag in _KEEPING_TAGS
        ]
        # What a prune takes away is decided in one pass before anything is read out of the chain: a
        # prune is a row like any other and its targets are older than it, but the entry it hides may sit
        # anywhere below, so the answer has to be known before the first message is handed over.
        withheld = {
            target for _entry, _turn, record in records if isinstance(record, Withheld) for target in record.targets
        }
        # An aside is in the log and not in the conversation, and both halves of that are rows: the one
        # its turn ended with says which kind of turn it was, and a person keeping it later writes one
        # naming it. Read in the same pass as the prune, for the same reason (E163).
        aside = {
            turn for _entry, turn, record in records if isinstance(record, TurnEnded) and record.kind == TurnKind.ASIDE
        }
        kept = {record.turn for _entry, _turn, record in records if isinstance(record, Promoted)}
        apart = aside - kept
        messages = tuple(
            message
            for entry, turn, record in records
            if entry not in withheld and turn not in apart and (message := as_message(record)) is not None
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

        Read as the session's own column, because a fact about a session is not a fact about a branch of
        it: a conversation that disclosed a group has disclosed it, and a
        branch that grew from before the disclosure is still part of that conversation.

        The fold starts at the newest snapshot this build can read and takes the rows after it, so a
        conversation with hundreds of settings behind it costs what has been written since the last
        fold rather than everything ever written (E177). With no snapshot to start from, that is every
        state row — the same answer, and slower.

        The fold is a dict built in arrival order, so a later entry overwrites an earlier one and no
        reader has to know that a key was written twice. What a setting *means* is the writer's
        business: this answers text, and a caller that wants a value parses it.
        """
        async with self._lock:
            return await self._folded_state(self._connection(), session)

    async def _folded_state(self, db: aiosqlite.Connection, session: str) -> dict[str, str]:
        """The settings fold, on a connection the caller already holds."""
        snapshot = await self._latest(db, session)
        rows = await self._all(
            db,
            "SELECT payload FROM entry WHERE session = ? AND tag = ? AND seq > ? ORDER BY seq",
            (session, STATE_CHANGE_TAG, 0 if snapshot is None else snapshot.seq),
        )
        folded: dict[str, str] = {} if snapshot is None else dict(snapshot.settings)
        for (payload,) in rows:
            record = msgspec.json.decode(str(payload), type=StateChange)
            folded[record.key] = record.value
        return folded

    @override
    async def shape(self, session: str) -> Shape:
        """Heads and roots, both in arrival order so two reads of an unchanged log agree.

        Read the way the settings are: the snapshot's forest, then the rows after it. A row that arrives
        is a head, and the entry it continues stops being one — so the fold is a walk over the new rows
        and never a walk over the whole session (E177).
        """
        async with self._lock:
            return await self._folded_shape(self._connection(), session)

    async def _folded_shape(self, db: aiosqlite.Connection, session: str) -> Shape:
        """The forest fold, on a connection the caller already holds."""
        snapshot = await self._latest(db, session)
        heads: list[str] = [] if snapshot is None else list(snapshot.heads)
        roots: list[str] = [] if snapshot is None else list(snapshot.roots)
        rows = await self._all(
            db,
            "SELECT id, parent FROM entry WHERE session = ? AND seq > ? ORDER BY seq",
            (session, 0 if snapshot is None else snapshot.seq),
        )
        headless = set(heads)
        for row, parent in rows:
            if parent is not None and f"e{parent}" in headless:
                headless.discard(f"e{parent}")
                heads.remove(f"e{parent}")
            heads.append(f"e{row}")
            headless.add(f"e{row}")
            if parent is None:
                roots.append(f"e{row}")
        return Shape(heads=tuple(heads), roots=tuple(roots))

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

        ``MIN(id)``, because a sequence number is per session — every session's first
        entry is number one — so only the row number is comparable across sessions.

        A conversation somebody deleted is not here: the row that says so is in its own chain, and this
        is the one reader that acts on it. Everything it said is still in the file, which is what makes
        "gone from the list" a fact about the list.
        """
        rows = await self._read_all(
            "SELECT session FROM entry WHERE session NOT IN (SELECT session FROM entry WHERE tag = ?) "
            "GROUP BY session ORDER BY MIN(id)",
            (DELETED_TAG,),
        )
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

    # -- the snapshot beside the log ------------------------------------------------

    @override
    async def latest(self, session: str) -> Snapshot | None:
        """The newest snapshot of this session this build wrote and can read, or ``None``.

        Two questions, asked in one query and one decode: the version the row was written with has to be
        this build's, and the projector recorded inside it has to be this build's. Either failing is
        "there is no snapshot here" — the fold then starts from the root, which is slower and the same
        answer (E177).
        """
        async with self._lock:
            return await self._latest(self._connection(), session)

    async def _latest(self, db: aiosqlite.Connection, session: str) -> Snapshot | None:
        """The newest readable snapshot, on a connection the caller already holds."""
        row = await self._one(
            db,
            "SELECT payload FROM checkpoint WHERE session = ? AND version = ? ORDER BY seq DESC LIMIT 1",
            (session, CHECKPOINTS),
        )
        if row is None:
            return None
        try:
            snapshot = msgspec.json.decode(str(row[0]), type=Snapshot)
        except msgspec.DecodeError:
            return None
        return None if snapshot.projector != PROJECTOR else snapshot

    @override
    async def put(self, session: str, snapshot: Snapshot) -> None:
        """Keep this snapshot, and let the oldest ones of this session go beyond the handful kept."""
        async with self._lock:
            await self._keep(self._connection(), session, snapshot)

    async def _keep(self, db: aiosqlite.Connection, session: str, snapshot: Snapshot) -> None:
        """Write one snapshot and trim the row, on a connection the caller already holds."""
        await self._run(
            db, _PUT_CHECKPOINT, (session, snapshot.seq, CHECKPOINTS, msgspec.json.encode(snapshot).decode())
        )
        await self._run(db, _TRIM_CHECKPOINTS, (session, session))

    @override
    async def snapshot(self, session: str) -> Snapshot:
        """Fold this conversation and keep the fold, with no append able to interleave.

        The whole thing is one turn of the lock, because that is what makes the recorded row the row the
        fold reached: a row appended between the fold and the write would be in the fold and before the
        ``seq``, and the next fold would apply it twice — which settings survive and a forest does not.
        """
        async with self._lock:
            db = self._connection()
            settings = await self._folded_state(db, session)
            shape = await self._folded_shape(db, session)
            top = await self._one(db, "SELECT MAX(seq) FROM entry WHERE session = ?", (session,))
            snapshot = Snapshot(
                seq=0 if top is None or top[0] is None else int(top[0]),
                projector=PROJECTOR,
                settings=settings,
                heads=shape.heads,
                roots=shape.roots,
            )
            if snapshot.seq:
                await self._keep(db, session, snapshot)
            return snapshot

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

    @staticmethod
    async def _run(db: aiosqlite.Connection, sql: str, params: _Params) -> None:
        """Run one statement whose rows nobody wants, on a connection the caller already holds."""
        _ = await db.execute(sql, params)

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

    @override
    async def latest(self, session: str) -> Snapshot | None:
        """The newest snapshot of this session this build can vouch for, or ``None``."""
        return await self._backend.latest(session)

    @override
    async def put(self, session: str, snapshot: Snapshot) -> None:
        """Keep one more snapshot of this session beside its log."""
        await self._backend.put(session, snapshot)

    @override
    async def snapshot(self, session: str) -> Snapshot:
        """Fold this session now and keep the fold, as the backend folds it."""
        return await self._backend.snapshot(session)


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
#: A table, because the tag is already a column: ``msgspec`` refuses a union whose members disagree
#: about the name of their tag field, and these do — a message is tagged ``role`` while
#: every other kind is tagged ``type`` — so reading one row is a lookup and no JSON guessing. A tag
#: nothing here knows is read as a message, which is the only other thing a row could be.
_RECORDS: Mapping[str, type[AnyRecord]] = {
    STATE_CHANGE_TAG: StateChange,
    SUMMARY_TAG: Summary,
    NOTICE_TAG: Notice,
    ATTEMPT_TAG: Attempt,
    WITHHELD_TAG: Withheld,
    EFFECT_TAG: Effect,
    CONSENT_TAG: Consent,
    USAGE_TAG: Usage,
    TURN_ENDED_TAG: TurnEnded,
    PROMOTED_TAG: Promoted,
    DELETED_TAG: Deleted,
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
