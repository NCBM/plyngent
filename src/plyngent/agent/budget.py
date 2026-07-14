from __future__ import annotations

DEFAULT_TOOL_RESULT_MAX_CHARS = 32_000


def truncate_tool_result(text: str, max_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS) -> str:
    """Cap tool output so huge dumps do not flood model context."""
    if max_chars < 1:
        return text
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}\n...[truncated {omitted} characters]"
