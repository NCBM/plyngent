from __future__ import annotations

import asyncio
import re
import sys

from click import style

from plyngent.agent import ToolTag, tool
from plyngent.prompting import get_prompt_backend, read_line_with_timeout, run_prompt_async

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _wait_prompt(duration: int, *, reason: str | None) -> str:
    """Two-line wait prompt: status line, then the optional-reason input prompt.

    ``read_line_with_timeout`` writes the prompt verbatim to stdout (no click
    echo ANSI stripping), so styling is applied here and stripped when stdout
    is not a TTY. On a terminal: cyan status, dimmed detail, yellow input
    prompt; plain text otherwise (tests, redirected output).
    """
    status = style(f"Waiting for {duration}s", fg="cyan")
    if reason:
        status += style(f" ({reason})", fg="bright_black")
    status += style(". Press Enter to disturb.", fg="bright_black")
    input_prompt = style("Reason (optional): ", fg="yellow")
    rendered = f"{status}\n{input_prompt}"
    if not sys.stdout.isatty():
        return _ANSI_ESCAPE_RE.sub("", rendered)
    return rendered


@tool(name="wait", tags=ToolTag.LOCAL | ToolTag.READ_ONLY)
async def wait(duration: int, *, reason: str | None = None) -> str:
    """Wait ``duration`` seconds before continuing.

    Interactive sessions show a two-line prompt: a status line, then an
    optional-reason input — pressing Enter (optionally after typing a reason)
    "disturbs" the wait so the turn continues immediately. Non-interactive runs
    simply sleep the full duration.
    """
    if duration < 0:
        return "error: duration must not be negative"
    if duration == 0:
        return "waited 0s"
    backend = get_prompt_backend()
    if not backend.is_interactive():
        await asyncio.sleep(duration)
        return f"waited {duration}s"
    prompt = _wait_prompt(duration, reason=reason)
    line = await run_prompt_async(read_line_with_timeout, prompt, float(duration))
    if line is None:
        return f"waited {duration}s"
    text = line.strip()
    if text:
        return f"disturbed by user: {text}"
    return "disturbed by user (no reason)"
