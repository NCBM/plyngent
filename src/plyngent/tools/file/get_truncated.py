"""``get_truncated`` tool: resume truncated tool results via a truncate token."""

from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.agent.budget import DEFAULT_TOOL_RESULT_MAX_CHARS
from plyngent.tools.file.read import read_file
from plyngent.tools.net.fetch import fetch
from plyngent.tools.truncate_token import decode_truncate_token, truncate_with_token


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def get_truncated(token: str, *, max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS) -> str:
    """Fetch the next chunk of a truncated result using its ``TRUNCATE_TOKEN``.

    Pass the token shown at the end of a truncated ``read_file``, ``fetch``, or
    earlier ``get_truncated`` result to keep reading without re-requesting the
    whole source or raising limits. Truncated output carries a fresh token, so
    chunks chain indefinitely.
    """
    parsed = decode_truncate_token(token)
    if parsed is None:
        return "error: invalid truncate token"
    if parsed.kind == "http":
        return await fetch.handler(parsed.location, offset=parsed.offset, max_chars=parsed.limit)
    text = await read_file.handler(parsed.location)
    if text.startswith("error:"):
        return text
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
    return chunk
