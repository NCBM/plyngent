from __future__ import annotations

from plyngent.agent import tool
from plyngent.prompting import NonInteractiveError, ask_async


@tool(name="ask_user_line")
async def ask_user(question: str, default: str = "") -> str:
    """Ask the human a free-form one-line question and return their answer.

    Always allows arbitrary text. Use for clarifying requirements, preferences,
    or any input that is not a fixed menu. Optional ``default`` is used if the
    user submits empty input (and in non-interactive mode when provided).
    """
    try:
        return await ask_async(question, default=default or None)
    except NonInteractiveError as exc:
        return f"error: {exc}"
