from __future__ import annotations

import contextlib
import os
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING

import click

from plyngent.agent import (
    CancelledEvent,
    ErrorEvent,
    MaxRoundsEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
    UsageEvent,
)
from plyngent.lmproto.openai_compatible.model import AssistantFunctionToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.agent import AgentEvent

_TOOL_RESULT_PREVIEW = 120
_TOOL_ARGS_PREVIEW = 80

# Process/session display flags (set from ReplState / slash).
_verbose_tool_results: ContextVar[bool] = ContextVar("verbose_tool_results", default=False)
_markdown_enabled: ContextVar[bool] = ContextVar("markdown_enabled", default=True)


def set_verbose_tool_results(enabled: bool) -> None:  # noqa: FBT001
    """Set whether tool results print in full (True) or as a short preview."""
    _ = _verbose_tool_results.set(enabled)


def get_verbose_tool_results() -> bool:
    return _verbose_tool_results.get()


def set_markdown_enabled(enabled: bool) -> None:  # noqa: FBT001
    """Enable or disable end-of-turn Rich markdown rendering."""
    _ = _markdown_enabled.set(enabled)


def get_markdown_enabled() -> bool:
    return _markdown_enabled.get()


def markdown_render_available() -> bool:
    """True when stdout is a TTY and plain mode is not forced via env."""
    if os.environ.get("PLYNGENT_PLAIN", "").strip() in {"1", "true", "yes", "on"}:
        return False
    try:
        return sys.stdout.isatty()
    except AttributeError, OSError, ValueError:
        return False


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _echo_stream(text: str) -> None:
    """Write without newline and flush so assistant text appears token-by-token."""
    click.echo(text, nl=False)
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _clear_streamed_lines(line_count: int) -> None:
    """Move cursor up and clear the streamed plain-text region (TTY only)."""
    if line_count <= 0:
        return
    # Clear current line, then each previous line of the streamed block.
    for _ in range(line_count):
        _ = sys.stdout.write("\r\033[2K\033[1A")
    _ = sys.stdout.write("\r\033[2K")
    with contextlib.suppress(OSError):
        _ = sys.stdout.flush()


def _line_count_for_clear(label: str, body: str) -> int:
    """Approximate terminal lines used by ``label + body`` for cursor erase."""
    if not body and not label:
        return 0
    text = label + body
    # +1 for trailing newline after the stream block in render_events.
    return text.count("\n") + 1


def print_markdown(text: str, *, label: str = "assistant: ") -> None:
    """Render *text* as markdown via Rich, with a cyan label."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    console = Console(file=sys.stdout, highlight=False)
    if label:
        console.print(Text(label, style="cyan"), end="")
    console.print(Markdown(text))


async def render_events(  # noqa: C901, PLR0912, PLR0915
    events: AsyncIterator[AgentEvent],
    *,
    verbose: bool | None = None,
    markdown: bool | None = None,
) -> None:
    """Print agent events to the terminal (text deltas stream as they arrive).

    When markdown is enabled and stdout is a TTY, the plain streamed assistant
    text is replaced at end-of-turn with a Rich markdown render.
    """
    show_full = get_verbose_tool_results() if verbose is None else verbose
    use_markdown = get_markdown_enabled() if markdown is None else markdown
    pretty = bool(use_markdown and markdown_render_available())

    printed_reasoning = False
    printed_text = False
    assistant_buf: list[str] = []
    # Tools/errors after assistant text: skip replace (would erase tool lines).
    interrupted_by_other = False

    async for event in events:
        if isinstance(event, ReasoningDeltaEvent):
            if not printed_reasoning:
                click.echo()
                click.secho("reasoning: ", fg="bright_black", nl=False)
                printed_reasoning = True
            _echo_stream(event.content)
        elif isinstance(event, TextDeltaEvent):
            if printed_reasoning and not printed_text:
                click.echo()
            if not printed_text:
                click.echo()
                click.secho("assistant: ", fg="cyan", nl=False)
                printed_text = True
            assistant_buf.append(event.content)
            _echo_stream(event.content)
        elif isinstance(event, ToolCallEvent):
            if printed_text:
                interrupted_by_other = True
            call = event.tool_call
            if isinstance(call, AssistantFunctionToolCall):
                args = _preview(call.function.arguments, _TOOL_ARGS_PREVIEW)
                click.secho(f"\n[tool] {call.function.name}({args})", fg="yellow")
            else:
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            if printed_text:
                interrupted_by_other = True
            content = event.message.content
            if show_full:
                click.secho(f"[tool ok]\n{content}", fg="magenta")
            else:
                preview = _preview(content, _TOOL_RESULT_PREVIEW)
                one_line = preview.replace("\n", " ")
                click.secho(f"[tool ok] {one_line}", fg="magenta")
        elif isinstance(event, ErrorEvent):
            if printed_text:
                interrupted_by_other = True
            suffix = ""
            if event.source:
                suffix += f" source={event.source}"
            if not event.retryable:
                suffix += " (fatal)"
            click.secho(f"\n[error]{suffix} {event.message}", fg="bright_red")
        elif isinstance(event, CancelledEvent):
            if printed_text:
                interrupted_by_other = True
            if event.reason:
                click.secho(f"\n[cancelled] {event.reason}", fg="yellow")
            else:
                click.secho("\n[cancelled]", fg="yellow")
        elif isinstance(event, MaxRoundsEvent):
            if printed_text:
                interrupted_by_other = True
            if event.continued:
                click.secho(
                    f"\n[max rounds {event.rounds} reached — continuing with a higher allowance]",
                    fg="yellow",
                )
            else:
                click.secho(f"\n[max rounds reached: {event.rounds}]", fg="red")
        elif isinstance(event, UsageEvent):
            _ = event
        else:
            # AssistantMessageEvent — text already shown via TextDeltaEvent.
            _ = event

    full_assistant = "".join(assistant_buf)
    if pretty and printed_text and full_assistant.strip() and not interrupted_by_other:
        # Replace plain stream with markdown (assistant block only).
        lines = _line_count_for_clear("assistant: ", full_assistant)
        # Also account for blank line before label when reasoning was shown.
        if printed_reasoning:
            lines += 1
        _clear_streamed_lines(lines)
        if printed_reasoning:
            # Reasoning already printed above; leave it and only re-print assistant.
            pass
        print_markdown(full_assistant, label="assistant: ")
        click.echo()
        return

    if printed_text or printed_reasoning:
        click.echo()
    click.echo()
