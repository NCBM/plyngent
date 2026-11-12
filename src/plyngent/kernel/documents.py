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

**How much fits is a property of the store.** :data:`KEEP_CHARS` is the threshold a tool asks this
object for rather than a number in a schema: a schema that changed with it would rewrite the prompt
prefix a provider caches (§10.5), and the executor does not know what a body means.
"""

from __future__ import annotations

import asyncio
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

#: The one version of this shape.
#:
#: There is no ladder yet: nothing has been released, so this script *is* the shape and a change before
#: the first release is a change to it. The version is stamped anyway, because a file written by this
#: build has to be able to say what it is to the build that comes after it — and because stamping it
#: costs nothing while a ladder invented for one version would be a promise nobody has kept yet.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS document (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session TEXT NOT NULL,
    summary TEXT NOT NULL,
    media TEXT NOT NULL,
    chars INTEGER NOT NULL,
    created INTEGER NOT NULL,
    content BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS document_session ON document (session);
PRAGMA user_version = 1;
"""

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
    _path: Path
    _db: aiosqlite.Connection | None
    _lock: asyncio.Lock

    def __init__(self, path: Path | str, *, keep_chars: int = KEEP_CHARS) -> None:
        super().__init__()
        self._path = Path(path)
        self.keep_chars = keep_chars
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
        except BaseException:
            self._db = None
            await db.close()
            raise

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
        how big the thing is to the reader that will ask for it.
        """
        data = content.encode("utf-8") if isinstance(content, str) else content
        chars = len(content)
        row = await self._execute(
            "INSERT INTO document (session, summary, media, chars, created, content) "
            "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
            (session, summary, media, chars, int(time.time()), data),
        )
        assert row is not None, "an insert that returns its row always has one"
        return Kept(id=_handle(row[0]), summary=summary, media=media, chars=chars)

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
