from __future__ import annotations

import asyncio

from plyngent.agent import ToolTag, tool
from plyngent.prompting import get_prompt_backend, read_line_with_timeout, run_prompt_async

WAIT_PROMPT = "Waiting for {duration}s. Press Enter to disturb. Feel free to ship any reason before pressing Enter. "


@tool(name="wait", tags=ToolTag.LOCAL | ToolTag.READ_ONLY)
async def wait(duration: int, *, reason: str | None = None) -> str:
    """Wait ``duration`` seconds before continuing.

    Interactive sessions show a line prompt: pressing Enter (optionally after
    typing a reason) "disturbs" the wait so the turn continues immediately.
    Non-interactive runs simply sleep the full duration.
    """
    if duration < 0:
        return "error: duration must not be negative"
    if duration == 0:
        return "waited 0s"
    backend = get_prompt_backend()
    if not backend.is_interactive():
        await asyncio.sleep(duration)
        return f"waited {duration}s"
    prompt = WAIT_PROMPT.format(duration=duration)
    if reason:
        prompt = (
            f"Waiting for {duration}s (reason: {reason}). Press Enter to disturb. "
            "Feel free to ship any reason before pressing Enter. "
        )
    line = await run_prompt_async(read_line_with_timeout, prompt, float(duration))
    if line is None:
        return f"waited {duration}s"
    text = line.strip()
    if text:
        return f"disturbed by user: {text}"
    return "disturbed by user (no reason)"
