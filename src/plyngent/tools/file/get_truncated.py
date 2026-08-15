"""``get_truncated`` tool: resume truncated tool results via a truncate token."""

from __future__ import annotations

from typing import cast

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.tools.file.read import line_range_label, read_file, read_raw_text
from plyngent.tools.net.fetch import fetch
from plyngent.tools.truncate_token import (
    TruncateToken,
    decode_truncate_token,
    get_remainder,
    truncate_with_token,
)


def _memory_chunk(parsed: TruncateToken, max_chars: int) -> str:
    """Serve one chunk from the in-memory remainder store (or an error)."""
    remainder = get_remainder(parsed.location)
    if remainder is None:
        return "error: truncate token expired (no longer in memory); re-run the tool"
    start = parsed.offset
    segment = remainder[start : start + max_chars]
    chunk, _ = truncate_with_token(
        segment,
        max_chars,
        kind="memory",
        location=parsed.location,
        offset=start,
        limit=max_chars,
        total_len=len(remainder),
    )
    return chunk


async def _numbered_continuation(text: str, location: str, char_offset: int, max_chars: int) -> str:
    """Resume a numbered read at the line containing *char_offset*.

    Delegates to ``read_file(with_lineno=True)`` so the continuation carries
    ``N|`` line numbers and marks the resumed lines readable for ``edit_lineno``.
    """
    pos = 0
    line = 0
    for ln in text.splitlines(keepends=True):
        if pos + len(ln) > char_offset:
            break
        pos += len(ln)
        line += 1
    return await read_file.handler(
        location,
        offset=line,
        with_lineno=True,
        max_chars=max_chars,
    )


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def get_truncated(token: str, *, max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS) -> str:
    """Fetch the next chunk of a truncated result using its ``truncate_token``.

    Pass the token shown at the end of a truncated ``read_file``, ``fetch``,
    ``run_command``, or earlier ``get_truncated`` result to keep reading without
    re-requesting the whole source or raising limits. Truncated output carries a
    fresh token, so chunks chain indefinitely. File chunks resume in the same
    view as the original read: numbered reads continue as numbered lines
    (editable via ``edit_lineno``), plain reads as raw text with a 1-based
    ``L{begin}-{end}`` range. Memory chunks (generic tool output) are served
    from a short-lived in-memory store that is forgotten when the agent exits.
    """
    parsed = decode_truncate_token(token)
    if parsed is None:
        return "error: invalid truncate token"
    if parsed.kind == "http":
        return await fetch.handler(parsed.location, offset=parsed.offset, max_chars=parsed.limit)
    if parsed.kind == "memory":
        return _memory_chunk(parsed, max_chars)
    text, err = read_raw_text(parsed.location)
    if err:
        return err
    text = cast("str", text)
    start = parsed.offset
    if parsed.numbered:
        return await _numbered_continuation(text, parsed.location, start, max_chars)
    segment = text[start : start + max_chars]
    chunk, _ = truncate_with_token(
        segment,
        max_chars,
        kind="file",
        location=parsed.location,
        offset=start,
        limit=max_chars,
        total_len=len(text),
    )
    content_len = len(chunk.split("\n[Truncated", 1)[0]) if "\n[Truncated" in chunk else len(chunk)
    header = line_range_label(text, start, start + content_len)
    return f"{header}\n{chunk}"
