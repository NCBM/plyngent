from __future__ import annotations

import contextlib
import os
import sys
from contextvars import ContextVar
from typing import TYPE_CHECKING, Literal

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

type StreamSource = Literal["reasoning", "assistant"]


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
    """Approximate terminal lines used by ``label\\n + body`` for cursor erase."""
    if not body and not label:
        return 0
    # Label is on its own line; body may contain newlines.
    text = f"{label}\n{body}" if label else body
    return text.count("\n") + 1


def print_markdown(text: str, *, label: str = "assistant:") -> None:
    """Render *text* as markdown via Rich; *label* on its own line when set."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text

    console = Console(file=sys.stdout, highlight=False)
    if label:
        console.print(Text(label, style="cyan"))
    console.print(Markdown(text))


def _flush_assistant_markdown(body: str, *, pretty: bool) -> None:
    """Replace the plain assistant stream with markdown when enabled."""
    if not body.strip():
        click.echo()
        return
    if pretty:
        lines = _line_count_for_clear("assistant:", body)
        _clear_streamed_lines(lines)
        print_markdown(body, label="assistant:")
        click.echo()
    else:
        click.echo()


async def render_events(  # noqa: C901, PLR0912, PLR0915
    events: AsyncIterator[AgentEvent],
    *,
    verbose: bool | None = None,
    markdown: bool | None = None,
) -> None:
    """Print agent events to the terminal (text deltas stream as they arrive).

    Assistant and reasoning each start on a new line after their label. When the
    content source changes (reasoning ↔ assistant, or tools/errors), the
    assistant markdown buffer is flushed so streams do not mix and Rich can
    re-render completed assistant segments.
    """
    show_full = get_verbose_tool_results() if verbose is None else verbose
    use_markdown = get_markdown_enabled() if markdown is None else markdown
    pretty = bool(use_markdown and markdown_render_available())

    source: StreamSource | None = None
    assistant_buf: list[str] = []
    printed_reasoning = False
    printed_assistant = False

    def flush_assistant() -> None:
        nonlocal source, assistant_buf, printed_assistant
        if source != "assistant" and not assistant_buf:
            return
        body = "".join(assistant_buf)
        assistant_buf = []
        if printed_assistant:
            _flush_assistant_markdown(body, pretty=pretty)
        printed_assistant = False
        if source == "assistant":
            source = None

    def begin_reasoning() -> None:
        nonlocal source, printed_reasoning
        if source == "reasoning":
            return
        if source == "assistant":
            flush_assistant()
        click.echo()
        click.secho("reasoning:", fg="bright_black")
        source = "reasoning"
        printed_reasoning = True

    def begin_assistant() -> None:
        nonlocal source, printed_assistant
        if source == "assistant":
            return
        if source == "reasoning":
            click.echo()  # end reasoning stream line
            source = None
        click.echo()
        click.secho("assistant:", fg="cyan")
        source = "assistant"
        printed_assistant = True

    async for event in events:
        if isinstance(event, ReasoningDeltaEvent):
            begin_reasoning()
            _echo_stream(event.content)
        elif isinstance(event, TextDeltaEvent):
            begin_assistant()
            assistant_buf.append(event.content)
            _echo_stream(event.content)
        elif isinstance(event, ToolCallEvent):
            flush_assistant()
            call = event.tool_call
            if isinstance(call, AssistantFunctionToolCall):
                args = _preview(call.function.arguments, _TOOL_ARGS_PREVIEW)
                click.secho(f"\n[tool] {call.function.name}({args})", fg="yellow")
            else:
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            flush_assistant()
            content = event.message.content
            if show_full:
                click.secho(f"[tool ok]\n{content}", fg="magenta")
            else:
                preview = _preview(content, _TOOL_RESULT_PREVIEW)
                one_line = preview.replace("\n", " ")
                click.secho(f"[tool ok] {one_line}", fg="magenta")
        elif isinstance(event, ErrorEvent):
            flush_assistant()
            suffix = ""
            if event.source:
                suffix += f" source={event.source}"
            if not event.retryable:
                suffix += " (fatal)"
            click.secho(f"\n[error]{suffix} {event.message}", fg="bright_red")
        elif isinstance(event, CancelledEvent):
            flush_assistant()
            if event.reason:
                click.secho(f"\n[cancelled] {event.reason}", fg="yellow")
            else:
                click.secho("\n[cancelled]", fg="yellow")
        elif isinstance(event, MaxRoundsEvent):
            flush_assistant()
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

    # End-of-turn: flush any open assistant segment; close reasoning stream.
    if assistant_buf or printed_assistant:
        flush_assistant()
    elif printed_reasoning:
        click.echo()
    click.echo()
