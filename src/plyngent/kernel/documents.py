"""Where content too long for the conversation is kept, and the handles that reach it back.

A result is read by a model whose context is finite, so a body it cannot take is neither cut in place
nor thrown away: it is kept here, and the result carries a handle plus the one line that stands in for
it. What a handle looks like is this module's business and nobody else's — an identifier promises only
that it locates the content, never that it is a number, a length, a name or a primary key.

**Its own file, beside the journal.** The journal is an append-only history whose schema is a ladder of
migrations; content has a different lifecycle (it can be expired, deduplicated, indexed for search) and
a different size profile (one body may be megabytes, one entry of the log never is). Keeping them apart
leaves the log a log, and lets the store be given retention, an index or a blob backend without a
migration that first has to know what a transcript is. The price is a second connection, a second lock,
two files per workspace and no transaction across both — and the price is paid in one direction only:
**the row is written before the result that carries the handle**, so a crash between the two leaves
content nobody refers to rather than a handle pointing at nothing.

**Every read is scoped to a session.** Content is as private as the conversation it was kept for, so
two sessions on one host do not see each other's documents by id, and an id that belongs to somebody
else is answered as if it did not exist.

**The same bytes are kept once, per session.** A row is (content, media) under the session that asked for
it, so a window read twice is one handle rather than two, and a handle is something a model can hold on
to. Sessions never share a row — that is the privacy rule above, read in the other direction.

**It is a cache, not an archive.** Every body here can be had again: a tool can be asked twice, a file
read twice, an export derived from the log again. So each session has a budget, and writing past it lets
the oldest bodies go — at the write, when the answer is known, rather than on a timer somebody would have
to trust. A handle to a body that has gone is not a broken handle: it is an id this session no longer
has, which is the answer a made-up id already gets.

**How much fits is a property of the store.** :data:`KEEP_CHARS` is the threshold a tool asks this
object for rather than a number in a schema: a schema that changed with it would rewrite the prompt
prefix a provider caches (§10.5), and the executor does not know what a body means.
:data:`MAX_TEXT_CHARS` and :data:`MAX_KEPT_BYTES` are the other end of the same idea — what this build
will hold at all — and :data:`SUMMARY_CHARS` keeps the line that stands in for content a line.

**There is no full-text index.** A search here is one handle at a time: the reader takes the content out
and matches it line by line, so an index would be a second copy of the same bytes with its own
consistency problem and nothing to answer that the content does not. Cross-document search is a
different call, and a different decision, which is when FTS5 becomes worth its price (§0.6, R10).
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

import aiosqlite
from composat import Component

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: The path that asks for a store which lasts only as long as the process — the in-memory database,
#: beside a journal that has no file either.
MEMORY_DOCUMENTS = ":memory:"

#: The media type of text a document tool produced, as opposed to a picture or anything else.
TEXT_MEDIA = "text/plain"

#: The pictures this build recognizes, by their leading bytes. A format is never guessed from a name:
#: what a provider is handed is the bytes, so the bytes are what says what they are.
_PICTURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

#: How much of a WebP header says so: four bytes of container, four of size, then the form type.
_WEBP_HEADER = 12

#: How much of a body goes into the conversation before it is kept instead.
#:
#: One number in one place, because it is a fact about this store: it is what "too long to send" means,
#: and every tool that produces a body asks the same question. ``main`` calls the same number
#: ``DEFAULT_TOOL_RESULT_MAX_CHARS``.
KEEP_CHARS = 32_000

#: The most text this build will keep, and the most it will hold in memory to search line by line.
#:
#: Four million characters is a long book and several times the largest page anybody fetches; past it the
#: answer is a window, not the whole thing. One number serves both questions because they are the same
#: question — how much text this build is willing to have in memory at once — and a second number for
#: searching would be a second opinion about the same trade.
MAX_TEXT_CHARS = 4_000_000

#: The most bytes it will keep: a picture, and a screenshot is the biggest thing that arrives.
#:
#: Bytes are not text in any window, so there is no size at which handing them over would be a favour —
#: which is why the binary limit is checked once, at the door, and not per read.
MAX_KEPT_BYTES = 8 * 1024 * 1024

#: How long a summary may be: it is the line that stands in for content, and a line is what it is.
SUMMARY_CHARS = 200

#: How much one session's kept bodies may take, and how many of them: the store is a cache, not an archive.
#:
#: A body is reconstructible — a tool can be asked again, a file read again, an export derived again from
#: the log — so the store keeps what a conversation is still likely to want and lets the oldest go. The
#: budget is in bytes (``length(content)`` in SQLite, which counts a blob as bytes and a string as
#: characters), because bytes are what the file costs. There is no timer anywhere: room is made when
#: something is written, which is the only moment the answer can be known (§0.6, R10).
SESSION_BUDGET = 64 * 1024 * 1024
SESSION_ROWS = 200


class TooBigError(Exception):
    """Content over this store's limit. The message is the sentence a tool shows, number and all."""


def refuses(content: str | bytes) -> str | None:
    """Why this build will not keep *content*, or ``None`` when it will.

    Asked by the writers before they get here (so a refusal can carry a hint) and again by
    :meth:`Documents.put` (so a writer that forgot cannot get past it): a limit that only the polite path
    honours is not a limit.
    """
    size = len(content)
    cap = MAX_TEXT_CHARS if isinstance(content, str) else MAX_KEPT_BYTES
    if size <= cap:
        return None
    unit = "characters" if isinstance(content, str) else "bytes"
    return f"{size} {unit} is more than this build will keep ({cap}): ask for a window of it instead"


#: The shape this build writes, and the first step of its ladder.
#:
#: ``sha256`` arrived with deduplication, so a file written before it is one column short and
#: :meth:`Documents._upgraded` adds it: the column is nullable, and a row from the older build keeps no
#: hash and is simply never matched by a later ``put`` — nothing is thrown away to gain an index. The
#: version is stamped once the shape is right rather than by this script, because a script that stamped it
#: would claim a shape that an older file does not have yet.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS document (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    sha256 BLOB,
    summary TEXT NOT NULL,
    media TEXT NOT NULL,
    chars INTEGER NOT NULL,
    created INTEGER NOT NULL,
    content BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS document_session ON document (session);
"""

#: The shape this build expects, which a file written by an earlier one has to be brought up to.
_VERSION = 2

#: What one document is read as, by everything that does not want its bytes.
_COLUMNS = "SELECT id, summary, media, chars FROM document"

#: How one statement is given its values.
type _Params = Sequence[object] | Mapping[str, object]


def documents_path(journal: Path | str) -> Path | str:
    """Where the documents of the journal at *journal* are kept: beside it, or nowhere.

    Beside, because the two are one workspace's: a person who moves a journal directory moves the
    content that was kept with it, and a journal with no file (``:memory:``) has no directory to put
    anything in.
    """
    if str(journal) == MEMORY_DOCUMENTS:
        return MEMORY_DOCUMENTS
    return Path(journal).parent / "documents.db"


def picture_media(data: bytes) -> str | None:
    """What picture format these bytes are, or ``None`` when they are not one this build knows.

    Shared by every reader of content — a file in the workspace, a body that came back over a network —
    because it is one question about one kind of bytes, and two tables would eventually disagree about
    one of them.
    """
    for signature, media in _PICTURES:
        if data.startswith(signature):
            return media
    if len(data) >= _WEBP_HEADER and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@dataclass(frozen=True, slots=True)
class Kept:
    """One content object, as anything that is not reading its bytes sees it."""

    id: str
    summary: str
    media: str
    chars: int


@dataclass(frozen=True, slots=True)
class Content:
    """A kept object with the bytes it holds: ``data`` is what the media type describes."""

    kept: Kept
    data: bytes


class Documents(Component):
    """One workspace's kept bodies: a handle in, content out, and never across a session.

    One connection and one lock, for the same reason the journal has them: every answer here is a step
    the rest of the kernel is waiting on, and one at a time is what a single connection can honestly
    promise.
    """

    keep_chars: int
    budget: int
    rows: int
    _path: Path
    _db: aiosqlite.Connection | None
    _lock: asyncio.Lock

    def __init__(
        self,
        path: Path | str,
        *,
        keep_chars: int = KEEP_CHARS,
        budget: int = SESSION_BUDGET,
        rows: int = SESSION_ROWS,
    ) -> None:
        super().__init__()
        self._path = Path(path)
        self.keep_chars = keep_chars
        self.budget = budget
        self.rows = rows
        self._db = None
        self._lock = asyncio.Lock()

    # -- the file ------------------------------------------------------------------

    @override
    async def load(self) -> None:
        """Open the store when the host comes up, so the first body that does not fit has somewhere to go."""
        await self.open()

    @override
    async def unload(self) -> None:
        """Close it — the host's last move, after every turn has been settled."""
        await self.close()

    async def open(self) -> None:
        """Make the directory, open the file, and bring the shape up to this build's."""
        if self._db is not None:
            return
        if not self._temporary:
            await asyncio.to_thread(self._path.parent.mkdir, parents=True, exist_ok=True)
        db = await aiosqlite.connect(str(self._path), isolation_level=None)
        self._db = db
        try:
            _ = await db.execute("PRAGMA journal_mode = WAL")
            _ = await db.execute("PRAGMA busy_timeout = 5000")
            _ = await db.executescript(_SCHEMA)
            await self._upgraded(db)
        except BaseException:
            self._db = None
            await db.close()
            raise

    async def _upgraded(self, db: aiosqlite.Connection) -> None:
        """Bring a file written by an earlier build up to this one: one column, one index, nothing lost.

        The first step of this store's ladder, and the reason it is a step rather than a fresh script: a
        person's ``documents.db`` predates the deduplication, and dropping their content to gain an index
        would be the store throwing away something it was asked to keep. A row from that build keeps no
        hash, which means it is never matched by a later ``put`` — it stays readable, and it is not part of
        the same-bytes-kept-once promise, which is a promise about what happens next.
        """
        async with db.execute("PRAGMA table_info(document)") as cursor:
            columns = {str(row[1]) for row in await cursor.fetchall()}
        if "sha256" not in columns:
            _ = await db.execute("ALTER TABLE document ADD COLUMN sha256 BLOB")
        _ = await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS document_same ON document (session, sha256, media)")
        _ = await db.execute(f"PRAGMA user_version = {_VERSION}")

    async def close(self) -> None:
        """Give the file back; a store already closed answers this by doing nothing."""
        db, self._db = self._db, None
        if db is not None:
            await db.close()

    @property
    def _temporary(self) -> bool:
        """Whether this store has no file at all, and therefore nothing to make before opening it."""
        return str(self._path) == MEMORY_DOCUMENTS

    # -- content -------------------------------------------------------------------

    async def put(self, session: str, content: str | bytes, *, summary: str, media: str = TEXT_MEDIA) -> Kept:
        """Keep *content* and answer the handle that reaches it.

        Text is stored as the bytes it encodes to, so the store has one column and one kind of value;
        ``chars`` counts characters for text and bytes for everything else, because it is there to say
        how big the thing is to the reader that will ask for it. Content over :func:`refuses`' limit is
        not kept, and a summary longer than :data:`SUMMARY_CHARS` is cut: both are the store keeping its
        own promise rather than trusting every writer to have asked.

        The same bytes of the same media in the same session are one row: a model that reads one window
        twice is handed the handle it already has, which is what makes a handle something it can keep
        rather than something that quietly multiplies. The row keeps the summary it was first given —
        a handle names content, and where it came from is the first place it came from. Sessions never
        share a row: content is as private as the conversation it was kept for.
        """
        refusal = refuses(content)
        if refusal is not None:
            raise TooBigError(refusal)
        data = content.encode("utf-8") if isinstance(content, str) else content
        chars = len(content)
        digest = hashlib.sha256(data).digest()
        same = await self._read(
            _COLUMNS + " WHERE session = ? AND sha256 = ? AND media = ?",
            (session, digest, media),
        )
        if same is not None:
            return _kept(same)
        line = _clipped(summary)
        row = await self._execute(
            "INSERT INTO document (session, sha256, summary, media, chars, created, content) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING id",
            (session, digest, line, media, chars, int(time.time()), data),
        )
        assert row is not None, "an insert that returns its row always has one"
        await self._made_room(session)
        return Kept(id=_handle(row[0]), summary=line, media=media, chars=chars)

    async def _made_room(self, session: str) -> None:
        """Let this session's oldest bodies go until it is inside its budget; the newest one always stays.

        Making room at the write rather than on a timer is the whole design: there is no clock to trust,
        no sweep to schedule, and the moment a body does not fit is exactly the moment to decide. The
        newest row is never dropped — a store that threw away what it had just been given would keep
        nothing at all for a body that is large on its own.
        """
        rows = await self._read_all(
            "SELECT id, length(content) FROM document WHERE session = ? ORDER BY id", (session,)
        )
        total = sum(int(str(size)) for _id, size in rows)
        dropped: list[object] = []
        while len(rows) > 1 and (total > self.budget or len(rows) > self.rows):
            oldest = rows.pop(0)
            dropped.append(oldest[0])
            total -= int(str(oldest[1]))
        if not dropped:
            return
        marks = ", ".join("?" * len(dropped))
        _ = await self._execute(f"DELETE FROM document WHERE id IN ({marks})", tuple(dropped))

    async def get(self, session: str, document_id: str) -> Content | None:
        """One document with its bytes, or ``None`` when this session has no such id."""
        row = _row_of(document_id)
        if row is None:
            return None
        found = await self._read(
            "SELECT id, summary, media, chars, content FROM document WHERE session = ? AND id = ?",
            (session, row),
        )
        if found is None:
            return None
        return Content(kept=_kept(found[:4]), data=_bytes(found[4]))

    async def listing(self, session: str) -> tuple[Kept, ...]:
        """What this session has kept, oldest first — the answer to "what is there to read"."""
        rows = await self._read_all(_COLUMNS + " WHERE session = ? ORDER BY id", (session,))
        return tuple(_kept(row) for row in rows)

    # -- the connection ------------------------------------------------------------

    def _connection(self) -> aiosqlite.Connection:
        """The open connection, or a refusal: a store used before it is up is a bug, not a state."""
        db = self._db
        if db is None:
            msg = f"the documents at {self._path} are not open"
            raise RuntimeError(msg)
        return db

    async def _execute(self, sql: str, params: _Params) -> tuple[object, ...] | None:
        """Run one statement under the lock, and answer the row it returned if it returned one."""
        async with self._lock, self._connection().execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return None if row is None else tuple(row)

    async def _read(self, sql: str, params: _Params) -> tuple[object, ...] | None:
        """One row of a query, under the lock that keeps this connection's statements in order."""
        async with self._lock, self._connection().execute(sql, params) as cursor:
            row = await cursor.fetchone()
        return None if row is None else tuple(row)

    async def _read_all(self, sql: str, params: _Params) -> list[tuple[object, ...]]:
        """Every row of a query, under the same lock."""
        async with self._lock, self._connection().execute(sql, params) as cursor:
            return [tuple(row) for row in await cursor.fetchall()]


def _handle(row: object) -> str:
    """The name of the document in one row. Opaque outside this module, and stable across a restart."""
    return f"d{row}"


def _clipped(summary: str) -> str:
    """A summary as the one line it is: cut where it stops being a line, and marked where it was cut."""
    return summary if len(summary) <= SUMMARY_CHARS else summary[: SUMMARY_CHARS - 1] + "…"


def _row_of(document_id: str) -> int | None:
    """The row a handle was minted from, or ``None`` when it names nothing this store made."""
    prefix, digits = document_id[:1], document_id[1:]
    if prefix != "d" or not (digits.isascii() and digits.isdigit()):
        return None
    return int(digits)


def _kept(row: Sequence[object]) -> Kept:
    """One row's four describing columns as the value everything outside reads."""
    row_id, summary, media, chars = row
    return Kept(id=_handle(row_id), summary=str(summary), media=str(media), chars=int(str(chars)))


def _bytes(value: object) -> bytes:
    """A content column as bytes: a blob is what this table stores, and nothing else is expected."""
    assert isinstance(value, bytes), "the content column is a blob"
    return value
