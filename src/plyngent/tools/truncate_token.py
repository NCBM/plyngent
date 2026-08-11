"""Truncate-token cursors for resuming truncated tool results.

Every truncation site (the agent loop's generic tool-result cap,
``run_command`` / ``run_command_batch`` per-stream caps, ``read_file``,
``fetch``, and request-time compact shrinks) appends the same compact marker
when output is cut short::

    [Truncated (32000 chars max.; 500 omitted). truncate_token=...]

The model passes the token to ``get_truncated`` to read the next chunk without
re-requesting the whole source or raising limits. Tokens chain: each truncated
chunk carries a fresh token, so ``get_truncated`` keeps resuming until the
source is exhausted (then no marker is emitted).

Resumable sources (``file`` / ``http``) are re-read by ``location`` +
``offset``. Arbitrary tool output (run_command stdout, todo renders, …) has no
resumable source, so its remainder lives in a short-lived in-memory store
(``kind="memory"``) that is forgotten when the agent process exits — the model
must re-run the tool after that.
"""

from __future__ import annotations

import base64
import uuid
from typing import Literal

import msgspec

type TruncateKind = Literal["file", "http", "memory"]


class TruncateToken(msgspec.Struct, frozen=True):
    """Opaque cursor into a truncated source.

    ``offset``/``limit`` are in characters (http bodies, memory remainders) or
    lines (files is line-based via ``read_file``); ``get_truncated`` interprets
    by ``kind``. For ``memory``, ``location`` is a key into the in-memory
    remainder store and ``offset``/``limit`` index characters of that stored
    remainder.
    """

    kind: TruncateKind
    location: str
    offset: int
    limit: int


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


def truncate_generic(text: str, max_chars: int) -> str:
    """Cap arbitrary tool output; embed a memory truncate token (chainable).

    The full remainder is kept in the in-memory store so ``get_truncated`` can
    resume it. Returns ``text`` unchanged (no marker) when it fits within
    ``max_chars``.
    """
    if max_chars < 1 or len(text) <= max_chars:
        return text
    key = store_remainder(text)
    bounded, _ = truncate_with_token(
        text,
        max_chars,
        kind="memory",
        location=key,
        offset=0,
        limit=max_chars,
        total_len=len(text),
    )
    return bounded
