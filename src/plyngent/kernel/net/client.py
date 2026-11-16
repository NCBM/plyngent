"""One HTTP request, executed hop by hop, with a gate that may refuse each of them.

The engine knows how to follow a redirect and how to stop reading a body; it does not know who may ask,
whether an address is private, or what a person would say. That is the whole separation: **the engine
does not know there are people** — it calls ``gate`` before each connection, and a gate that answers
"no" ends the request there (§0.9). The same reason keeps it free of ``Env``: one call is one request,
and the handler that owns the call already runs under the turn's own scope.

Two decisions are this module's:

* **the verb is judged per hop, not per call.** A ``303`` turns a ``POST`` into a ``GET``, so the gate is
  handed the line that is about to be sent — asking about the method the call *named* would be asking
  about a request nobody is going to make. The body is dropped at the same moment, for the same reason.
* **a bad response is not a failure.** A ``404`` is the answer to the question that was asked: it comes
  back as a result with its status, and only "it never got there" is :class:`Stopped`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

import niquests

from plyngent.kernel.documents import picture_media
from plyngent.kernel.failure import Failure
from plyngent.kernel.net import policy
from plyngent.kernel.toolresult import DenialKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from typing import Any

#: The statuses that send a request somewhere else.
REDIRECT_STATUS: frozenset[int] = frozenset({301, 302, 303, 307, 308})

#: Content types that are text whatever the bytes look like, and ones that are not whatever they do.
_TEXT_HINTS: tuple[str, ...] = (
    "+json",
    "+xml",
    "application/graphql",
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/problem+json",
    "application/x-www-form-urlencoded",
    "application/xhtml",
    "application/xml",
    "text/",
)
_BINARY_HINTS: tuple[str, ...] = ("audio/", "image/", "octet-stream", "video/")

#: How much of a body decides whether it is text at all: enough to meet the first NUL byte.
_SNIFF_BYTES = 512

#: How much of a body is read at a time, so that a cap applies before the whole page is in memory.
_CHUNK_BYTES = 65_536


class BodyKind(StrEnum):
    """What the bytes came back as, in the four answers a reader acts on differently.

    ``EMPTY`` is its own kind rather than "text of length zero" because a ``HEAD`` and a ``304`` really
    have nothing to read, and a summary that said "0 bytes text" would be promising an empty page.
    """

    TEXT = "text"
    EMPTY = "empty"
    IMAGE = "image"
    BINARY = "binary"


class Fix(StrEnum):
    """Which argument would have let the request finish, as the engine names it.

    Data, not a hint: the engine may not name a tool (``Hint`` carries one, and naming tools is the tool
    layer's business, §10.3). It says *what* would have changed the outcome and the caller says *how*.
    """

    ALLOW_DOWNGRADE = "allow_http_downgrade"
    MORE_REDIRECTS = "max_redirects"


@dataclass(frozen=True, slots=True)
class Leg:
    """One hop's request line: where it goes, with which verb, and whether it still carries a body.

    A gate is asked per leg rather than per call because a redirect is a second destination — and
    because the verb and the body can change on the way, so what a gate judges is the line about to be
    sent and never the one the call wrote down.
    """

    url: policy.ParsedUrl
    method: str
    has_body: bool


@dataclass(frozen=True, slots=True)
class Stopped:
    """Why the request did not finish: the kind of trouble, the sentence, and what would change it.

    The same three things a refused command carries, and for the same reason: the kind is what decides
    status and retry (:data:`~plyngent.kernel.toolresult.OUTCOMES`), the sentence is what a reader is
    told, and neither is derived from the other. ``fix`` names the argument that would have let the
    request finish — data, because a hint carries a tool's name and the engine may not name one.
    """

    failure: Failure
    message: str
    fix: Fix | None = None
    #: Why, machine-readably, when the failure is a denial — the same reason
    #: :class:`~plyngent.kernel.process.Refused` carries one (§0.15, E118).
    denial: DenialKind | None = None


@dataclass(frozen=True, slots=True)
class Request:
    """One request, as the tool layer read it from a call.

    Everything already normalized: the method is one of the five, the headers are the ones that will be
    sent, the body is encoded. The engine re-parses the URL per hop anyway, because a redirect hands it a
    string and because the same parse is what a gate is asked about.
    """

    method: str
    url: str
    headers: Mapping[str, str]
    body: bytes | None = None
    timeout_seconds: float = policy.DEFAULT_TIMEOUT_SECONDS
    max_bytes: int = policy.DEFAULT_MAX_BYTES
    max_redirects: int = policy.DEFAULT_MAX_REDIRECTS
    follow_redirects: bool = True
    allow_http_downgrade: bool = False


@dataclass(frozen=True, slots=True)
class Fetched:
    """What came back: the response line, the bytes, and what it took to get them."""

    status: int
    method: str
    url: policy.ParsedUrl
    content_type: str
    body: bytes
    kind: BodyKind
    truncated: bool
    redirects: int
    notes: tuple[str, ...] = ()

    def as_text(self) -> str:
        """The body as the text its content type says it is, with what cannot be decoded replaced."""
        return decode_text(self.body, self.content_type)


#: One hop, asked whether it may be made: ``None`` is "go on", and a :class:`Stopped` is the answer.
type Gate = Callable[[Leg], Awaitable[Stopped | None]]


async def fetch(request: Request, *, gate: Gate) -> Fetched | Stopped:
    """Make *request*, asking ``gate`` about every hop before it is connected."""
    current = policy.parse_url(request.url)
    if isinstance(current, policy.Unreadable):
        return Stopped(Failure.ARGS, current.message)
    method, body = request.method, request.body
    notes: list[str] = []
    redirects = 0
    async with niquests.AsyncSession() as session:
        while True:
            stopped = await gate(Leg(url=current, method=method, has_body=body is not None))
            if stopped is not None:
                return stopped
            try:
                response = await session.request(
                    method,
                    current.url,
                    headers=dict(request.headers),
                    data=body,
                    timeout=float(request.timeout_seconds),
                    allow_redirects=False,
                    stream=True,
                )
            except niquests.RequestException as exc:
                return _trouble(exc, url=current.url, seconds=request.timeout_seconds)
            status = int(getattr(response, "status_code", 0) or 0)
            content_type = _content_type(response)
            if request.follow_redirects and status in REDIRECT_STATUS:
                if redirects >= request.max_redirects:
                    await _close(response)
                    return Stopped(
                        Failure.LIMIT,
                        f"more than {request.max_redirects} {_redirects(request.max_redirects)} from {request.url}",
                        fix=Fix.MORE_REDIRECTS,
                    )
                following = _next_hop(current, response, notes=notes, allow_downgrade=request.allow_http_downgrade)
                await _close(response)
                if isinstance(following, Stopped):
                    return following
                redirects += 1
                current = following
                method, body = _after_redirect(status, method, body)
                continue
            data, truncated = await _read_capped(response, max_bytes=request.max_bytes)
            await _close(response)
            return Fetched(
                status=status,
                method=method,
                url=current,
                content_type=content_type,
                body=data,
                kind=_kind(data, content_type),
                truncated=truncated,
                redirects=redirects,
                notes=tuple(dict.fromkeys(notes)),
            )


def decode_text(data: bytes, content_type: str) -> str:
    """*data* as text, in the charset the content type declares and with the rest replaced.

    A charset this build does not know falls back to utf-8 rather than refusing the body: what came back
    is still worth reading, and the alternative is a page of bytes nobody can use.
    """
    charset = "utf-8"
    lowered = content_type.lower()
    if "charset=" in lowered:
        charset = lowered.split("charset=", 1)[1].split(";")[0].strip().strip('"') or charset
    try:
        return data.decode(charset, errors="replace")
    except LookupError:
        return data.decode("utf-8", errors="replace")


def _next_hop(
    current: policy.ParsedUrl, response: Any, *, notes: list[str], allow_downgrade: bool
) -> policy.ParsedUrl | Stopped:
    """Where *response* sends the request next, or why that hop may not be made."""
    location = _header(response, "location")
    if location is None:
        return Stopped(Failure.PROTOCOL, f"{current.url} answered {_status(response)} with no Location to follow")
    following = policy.resolve_location(current, location)
    if isinstance(following, policy.Unreadable):
        return Stopped(
            Failure.PROTOCOL, f"{current.url} redirected to something that is not a URL: {following.message}"
        )
    if policy.is_downgrade(current, following):
        if not allow_downgrade:
            return Stopped(
                Failure.DENIED,
                f"{current.url} redirects to {following.url}, which is cleartext http, and "
                "allow_http_downgrade is not set",
                fix=Fix.ALLOW_DOWNGRADE,
                denial=DenialKind.RULE,
            )
        notes.append(f"https-to-http redirect to {following.url}")
    elif following.scheme == "http":
        notes.append(f"cleartext HTTP at {following.url}")
    return following


def _after_redirect(status: int, method: str, body: bytes | None) -> tuple[str, bytes | None]:
    """The verb and the body a redirect leaves the next hop with, the way a browser reads one."""
    if status in {302, 303} and method != "GET":
        return "GET", None
    return method, body


def _trouble(exc: niquests.RequestException, *, url: str, seconds: float) -> Stopped:
    """One transport failure as the kind of trouble it is.

    A timeout is its own kind because the same call may well work later, and everything else — refused,
    reset, unresolvable transport, a protocol the session could speak — is reported as it said it, which
    is the only wording nobody has to guess at.
    """
    if isinstance(exc, niquests.Timeout):
        return Stopped(Failure.TIMEOUT, f"{seconds} seconds passed and {url} had not answered")
    return Stopped(Failure.PROTOCOL, f"{url} could not be reached: {exc}")


def _kind(data: bytes, content_type: str) -> BodyKind:
    """What these bytes are, decided by the signature first and the content type second."""
    if not data:
        return BodyKind.EMPTY
    if picture_media(data) is not None:
        return BodyKind.IMAGE
    return BodyKind.TEXT if _looks_text(content_type, data) else BodyKind.BINARY


def _looks_text(content_type: str, sample: bytes) -> bool:
    """Whether a body of this type, whose first bytes look like this, can be read as text."""
    lowered = content_type.lower()
    if any(hint in lowered for hint in _TEXT_HINTS):
        return True
    if b"\x00" in sample[:_SNIFF_BYTES]:
        return False
    return not any(hint in lowered for hint in _BINARY_HINTS)


def _content_type(response: Any) -> str:
    """The response's content type, or nothing when it did not say."""
    return _header(response, "content-type") or ""


def _header(response: Any, name: str) -> str | None:
    """One response header, whichever way the session spells it."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for spelling in (name, name.title()):
        found = headers.get(spelling)
        if found is not None:
            return str(found)
    return None


def _status(response: Any) -> int:
    return int(getattr(response, "status_code", 0) or 0)


async def _read_capped(response: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    """At most *max_bytes* of the body, and whether there was more.

    Streamed rather than buffered whole: the cap exists so that one call cannot fill this process's
    memory with somebody else's page, and a body read first and cut afterwards would pay that cost before
    the cap ever applied.
    """
    stream: Any = response.iter_content(chunk_size=_CHUNK_BYTES)
    if hasattr(stream, "__await__"):
        stream = await stream
    if hasattr(stream, "__aiter__"):
        return await _drained_async(stream, max_bytes=max_bytes)
    return _drained(stream, max_bytes=max_bytes)


def _take(chunks: list[bytes], total: int, data: bytes, *, max_bytes: int) -> tuple[int, bool]:
    """Append one chunk under the cap, answering what was kept and whether the rest is being dropped."""
    room = max_bytes - total
    if room <= 0:
        return total, True
    if len(data) > room:
        chunks.append(data[:room])
        return max_bytes, True
    chunks.append(data)
    return total + len(data), False


async def _drained_async(stream: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    """Read an async stream until it ends or the cap is reached."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in stream:
        if not chunk:
            continue
        total, truncated = _take(chunks, total, bytes(chunk), max_bytes=max_bytes)
        if truncated:
            return b"".join(chunks), True
    return b"".join(chunks), False


def _drained(stream: Any, *, max_bytes: int) -> tuple[bytes, bool]:
    """Read a plain iterator the same way, for a session that hands one over instead."""
    chunks: list[bytes] = []
    total = 0
    for chunk in stream:
        if not chunk:
            continue
        total, truncated = _take(chunks, total, bytes(chunk), max_bytes=max_bytes)
        if truncated:
            return b"".join(chunks), True
    return b"".join(chunks), False


async def _close(response: Any) -> None:
    """Let go of a response, streaming or not."""
    closed = response.close()
    if hasattr(closed, "__await__"):
        await closed


def _redirects(count: int) -> str:
    """The noun a redirect budget is counted in, so a limit of one does not read like a bug."""
    return "redirect" if count == 1 else "redirects"
