from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.prompting import NonInteractiveError, ask_async
from plyngent.tools.chat.shape import first_error, string_error


@tool(name="ask_user_line", tags=ToolTag.LOCAL | ToolTag.READ_ONLY)
async def ask_user(question: str, default: str = "") -> str:
    """Ask the human a free-form one-line question and return their answer.

    Always allows arbitrary text. Use for clarifying requirements, preferences,
    or any input that is not a fixed menu. Optional ``default`` is used if the
    user submits empty input (and in non-interactive mode when provided).
    """
    shape_error = first_error(
        string_error("ask_user_line", "question", question),
        string_error("ask_user_line", "default", default),
    )
    if shape_error is not None:
        return f"error: {shape_error}"
    try:
        return await ask_async(question, default=default or None)
    except NonInteractiveError as exc:
        return f"error: {exc}"
