from __future__ import annotations

import asyncio
import re
import sys

from click import style

from plyngent.agent import ToolTag, tool
from plyngent.prompting import (
    PromptCancelledError,
    get_prompt_backend,
    read_line_with_timeout,
    run_cancellable_prompt_async,
)
from plyngent.tools.chat.shape import first_error, number_arg, string_error

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _wait_prompt(duration: float, *, reason: str | None) -> str:
    """Two-line wait prompt: status line, then the optional-reason input prompt.

    ``read_line_with_timeout`` writes the prompt verbatim to stdout (no click
    echo ANSI stripping), so styling is applied here and stripped when stdout
    is not a TTY. On a terminal: cyan status, dimmed detail, yellow input
    prompt; plain text otherwise (tests, redirected output).
    """
    status = style(f"Waiting for {duration:g}s", fg="cyan")
    if reason:
        status += style(f" ({reason})", fg="bright_black")
    status += style(". Press Enter to disturb.", fg="bright_black")
    input_prompt = style("Reason (optional): ", fg="yellow")
    rendered = f"{status}\n{input_prompt}"
    if not sys.stdout.isatty():
        return _ANSI_ESCAPE_RE.sub("", rendered)
    return rendered


async def _interactive_wait(seconds: float, reason: str | None) -> str:
    """Show the two-line wait prompt and report how the wait ended."""
    label = f"{seconds:g}"
    prompt = _wait_prompt(seconds, reason=reason)
    try:
        line = await run_cancellable_prompt_async(read_line_with_timeout, prompt, seconds)
    except PromptCancelledError:
        return "cancelled by user"
    if line is None:
        return f"waited {label}s"
    text = line.strip()
    if text:
        return f"disturbed by user: {text}"
    return "disturbed by user (no reason)"


@tool(name="wait", tags=ToolTag.LOCAL | ToolTag.READ_ONLY)
async def wait(duration: int, *, reason: str | None = None) -> str:
    """Wait ``duration`` seconds before continuing.

    Interactive sessions show a two-line prompt: a status line, then an
    optional-reason input — pressing Enter (optionally after typing a reason)
    "disturbs" the wait so the turn continues immediately, and Ctrl+C cancels
    the wait (the turn continues). Non-interactive runs simply sleep the full
    duration.
    """
    seconds, duration_error = number_arg("wait", "duration", duration, example="5")
    reason_error = None if reason is None else string_error("wait", "reason", reason)
    shape_error = first_error(duration_error, reason_error)
    if shape_error is not None:
        return f"error: {shape_error}"
    assert seconds is not None  # number_arg returns a value whenever it accepts
    if seconds < 0:
        return "error: duration must not be negative"
    if seconds == 0:
        return "waited 0s"
    backend = get_prompt_backend()
    if not backend.is_interactive():
        await asyncio.sleep(seconds)
        return f"waited {seconds:g}s"
    return await _interactive_wait(seconds, reason)
