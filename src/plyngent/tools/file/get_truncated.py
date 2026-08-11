"""``get_truncated`` tool: resume truncated tool results via a truncate token."""

from __future__ import annotations

from typing import cast

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.tools.file.read import line_range_label, read_raw_text
from plyngent.tools.net.fetch import fetch
from plyngent.tools.truncate_token import (
    decode_truncate_token,
    get_remainder,
    truncate_with_token,
)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def get_truncated(token: str, *, max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS) -> str:
    """Fetch the next chunk of a truncated result using its ``truncate_token``.

    Pass the token shown at the end of a truncated ``read_file``, ``fetch``,
    ``run_command``, or earlier ``get_truncated`` result to keep reading without
    re-requesting the whole source or raising limits. Truncated output carries a
    fresh token, so chunks chain indefinitely. File chunks start with a 1-based
    ``L{begin}-{end}`` line range like ``read_file``; memory chunks (generic
    tool output) are served from a short-lived in-memory store that is forgotten
    when the agent exits.
    """
    parsed = decode_truncate_token(token)
    if parsed is None:
        return "error: invalid truncate token"
    if parsed.kind == "http":
        return await fetch.handler(parsed.location, offset=parsed.offset, max_chars=parsed.limit)
    if parsed.kind == "memory":
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
    text, err = read_raw_text(parsed.location)
    if err:
        return err
    text = cast("str", text)
    start = parsed.offset
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
