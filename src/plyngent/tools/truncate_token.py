"""Truncate-token cursors for resuming truncated reads.

Two sources use a truncate token:

- ``read_file`` caps a slice with a ``file`` token; the remainder is re-read from
  disk by ``location`` + ``offset``, so it survives for as long as the file does
  (and ``numbered`` resumes the ``N|`` view for ``edit_lineno``).
- A user ``@path`` mention inlines a file with a ``memory`` token. The mention
  carries text into a user turn rather than a tool message, so the remainder
  lives in a short-lived in-memory store that is cleared when the agent process
  exits.

Tool results no longer use this module. A result is stored whole in its own
message, and the request-prep pass trims the request copy with a marker naming a
document (``read_document`` / ``search_document``; see ``tools/documents.py``).

Either way the model passes the token to ``get_truncated`` to read the next
chunk without re-requesting the whole source.
"""

from __future__ import annotations

import base64
import uuid
from typing import Literal

import msgspec

type TruncateKind = Literal["file", "memory"]


class TruncateToken(msgspec.Struct, frozen=True):
    """Opaque cursor into a truncated source.

    ``offset``/``limit`` are in characters (memory remainders) or lines (files
    is line-based via ``read_file``); ``get_truncated`` interprets by ``kind``.
    For ``memory``, ``location`` is a key into the in-memory remainder store and
    ``offset``/``limit`` index characters of that stored remainder. ``numbered``
    marks file tokens from ``read_file(with_lineno)``: the continuation resumes
    the numbered view instead of raw text.
    """

    kind: TruncateKind
    location: str
    offset: int
    limit: int
    numbered: bool = False


# In-memory remainders for generic (non-resumable) tool output. Process-lifetime
# only: forgotten when the agent exits. Single-user CLI → one process per agent.
_MEMORY_REMAINDERS: dict[str, str] = {}


def store_remainder(text: str) -> str:
    """Store truncated-away output; return its opaque store key."""
    key = f"mem-{uuid.uuid4().hex[:12]}"
    _MEMORY_REMAINDERS[key] = text
    return key


def get_remainder(key: str) -> str | None:
    """Fetch a stored remainder by key; None when expired/unknown."""
    return _MEMORY_REMAINDERS.get(key)


def encode_truncate_token(token: TruncateToken) -> str:
    """Base64url-encode a token so it is safe inside tool-result text."""
    payload = msgspec.json.encode(token)
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decode_truncate_token(text: str) -> TruncateToken | None:
    """Decode a token; None when malformed (model garbage or tampering)."""
    try:
        raw = base64.urlsafe_b64decode(text.encode("ascii"))
        token = msgspec.json.decode(raw, type=TruncateToken)
    except ValueError, msgspec.DecodeError:
        return None
    if token.offset < 0 or token.limit < 1:
        return None
    return token


def truncation_marker(max_chars: int, omitted: int, token: TruncateToken) -> str:
    """The single truncation hint: ``[Truncated (N chars max.; M omitted). truncate_token=...]``."""
    return f"\n[Truncated ({max_chars} chars max.; {omitted} omitted). truncate_token={encode_truncate_token(token)}]"


def truncate_with_token(
    text: str,
    max_chars: int,
    *,
    kind: TruncateKind,
    location: str,
    offset: int,
    limit: int,
    total_len: int,
    emit_more_after: bool = True,
) -> tuple[str, TruncateToken | None]:
    """Bound ``text`` (a slice of the full source starting at ``offset``) to ``max_chars``.

    Returns ``(bounded_text, next_token)``. A token is embedded whenever the
    returned chunk is cut (overflow) or, with ``emit_more_after`` (default),
    whenever more source content remains after the chunk — so ``get_truncated``
    chains through truncations. The token advances by the length actually
    returned (no content is skipped), and the marker is budgeted so the total
    returned length stays near ``max_chars`` (the agent loop's own cap never
    re-cuts it and never strips the token). ``omitted`` in the marker counts
    source characters not yet delivered. When nothing was cut and
    (``emit_more_after=False`` or the source ends), ``text`` is returned
    unchanged with no marker.
    """
    if max_chars < 1:
        return text, None
    if len(text) > max_chars:
        # First pass sizes the marker with an offset+max_chars token; re-encode
        # with the real cut so the cursor lands exactly where content stops.
        first = TruncateToken(kind=kind, location=location, offset=offset + max_chars, limit=limit)
        omitted = total_len - (offset + max_chars)
        cut = max_chars - len(truncation_marker(max_chars, omitted, first))
        if cut < 1:
            return text[:max_chars] + truncation_marker(max_chars, omitted, first), first
        token = TruncateToken(kind=kind, location=location, offset=offset + cut, limit=limit)
        return text[:cut] + truncation_marker(max_chars, omitted, token), token
    if not emit_more_after:
        return text, None
    more_after = offset + len(text) < total_len
    if not more_after:
        return text, None
    omitted = total_len - (offset + len(text))
    token = TruncateToken(kind=kind, location=location, offset=offset + len(text), limit=limit)
    return text + truncation_marker(max_chars, omitted, token), token


def truncate_to_memory(text: str, max_chars: int) -> tuple[str, TruncateToken | None]:
    """Cap ``text`` (a whole source) with a memory token over a stored copy.

    Nothing is stored when the text fits, so a caller that only needs the
    bounded text does not pay for a remainder. Returns ``(bounded_text, token)``
    where ``token`` is ``None`` when the source is exhausted. A user ``@path``
    mention is the caller: its inlined text has no re-readable source, so the
    remainder waits in memory until the process exits.
    """
    if max_chars < 1 or len(text) <= max_chars:
        return text, None
    key = store_remainder(text)
    return truncate_with_token(
        text,
        max_chars,
        kind="memory",
        location=key,
        offset=0,
        limit=max_chars,
        total_len=len(text),
    )
