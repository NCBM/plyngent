"""Truncate-token cursors for resuming truncated tool results.

``fetch`` and ``read_file`` append a ``[TRUNCATE_TOKEN: ...]`` marker when they
cut output short; the model passes the token to ``get_truncated`` to read the
next chunk without re-requesting the whole source or raising limits.
"""

from __future__ import annotations

import base64
from typing import Literal

import msgspec

type TruncateKind = Literal["file", "http"]


class TruncateToken(msgspec.Struct, frozen=True):
    """Opaque cursor into a truncated source.

    ``offset``/``limit`` are in characters (http bodies) or lines (files is
    line-based via ``read_file``); ``get_truncated`` interprets by ``kind``.
    """

    kind: TruncateKind
    location: str
    offset: int
    limit: int


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


def _truncated_marker(omitted: int, token: TruncateToken) -> str:
    return (
        f"\n...[truncated {omitted} chars; more via get_truncated token] "
        f"[TRUNCATE_TOKEN: {encode_truncate_token(token)}]"
    )


def truncate_with_token(
    text: str,
    max_chars: int,
    *,
    kind: TruncateKind,
    location: str,
    offset: int,
    limit: int,
    total_len: int,
) -> tuple[str, TruncateToken | None]:
    """Bound ``text`` to ``max_chars``; return (bounded_text, next_token).

    ``text`` is a slice of the full source starting at ``offset``; ``total_len``
    is the full source length. A token is embedded whenever more content remains
    after the returned chunk, so ``get_truncated`` chains through truncations
    and each chunk may itself carry a fresh token. The token advances by the
    length actually returned (no content is skipped), and the marker is budgeted
    so the total returned length stays near ``max_chars`` (the agent loop's own
    cap never re-cuts it and never strips the token).
    """
    if max_chars < 1:
        return text, None
    if len(text) > max_chars:
        omitted = len(text) - max_chars
        # First pass sizes the marker with an offset+max_chars token; re-encode
        # with the real cut so the cursor lands exactly where content stops.
        first = TruncateToken(kind=kind, location=location, offset=offset + max_chars, limit=limit)
        cut = max_chars - len(_truncated_marker(omitted, first))
        if cut < 1:
            return text[:max_chars] + _truncated_marker(omitted, first), first
        token = TruncateToken(kind=kind, location=location, offset=offset + cut, limit=limit)
        return text[:cut] + _truncated_marker(omitted, token), token
    more_after = offset + len(text) < total_len
    if not more_after:
        return text, None
    token = TruncateToken(kind=kind, location=location, offset=offset + len(text), limit=limit)
    marker = (
        f"\n...[more content available; use get_truncated with the token] "
        f"[TRUNCATE_TOKEN: {encode_truncate_token(token)}]"
    )
    return text + marker, token
