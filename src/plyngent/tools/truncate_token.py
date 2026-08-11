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
    and each chunk may itself carry a fresh token. The marker is budgeted so the
    total returned length stays at or under ``max_chars`` (the agent loop's own
    cap never re-cuts it and never strips the token).
    """
    if max_chars < 1:
        return text, None
    overflow = len(text) > max_chars
    more_after = offset + max_chars < total_len
    if not overflow and not more_after:
        return text, None
    token = TruncateToken(kind=kind, location=location, offset=offset + max_chars, limit=limit)
    tok_text = encode_truncate_token(token)
    if overflow:
        omitted = len(text) - max_chars
        marker = f"\n...[truncated {omitted} chars; more via get_truncated token] [TRUNCATE_TOKEN: {tok_text}]"
        cut = max_chars - len(marker)
        if cut < 1:
            # Marker longer than the budget: emit anyway so chaining never drops.
            return text[:max_chars] + marker, token
        return text[:cut] + marker, token
    marker = f"\n...[more content available; use get_truncated with the token] [TRUNCATE_TOKEN: {tok_text}]"
    return text + marker, token
