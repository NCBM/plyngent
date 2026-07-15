from __future__ import annotations

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

# Process/session display flag for tool result dumps (set from ReplState / slash).
_verbose_tool_results: ContextVar[bool] = ContextVar("verbose_tool_results", default=False)


def set_verbose_tool_results(enabled: bool) -> None:  # noqa: FBT001
    """Set whether tool results print in full (True) or as a short preview."""
    _ = _verbose_tool_results.set(enabled)


def get_verbose_tool_results() -> bool:
    return _verbose_tool_results.get()


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _echo_stream(text: str) -> None:
    """Write without newline and flush so assistant text appears token-by-token."""
    click.echo(text, nl=False)
    try:
        import sys

        _ = sys.stdout.flush()
    except OSError:
        pass


async def render_events(  # noqa: C901, PLR0912, PLR0915
    events: AsyncIterator[AgentEvent],
    *,
    verbose: bool | None = None,
) -> None:
    """Print agent events to the terminal (text deltas stream as they arrive)."""
    show_full = get_verbose_tool_results() if verbose is None else verbose
    printed_reasoning = False
    printed_text = False
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
            _echo_stream(event.content)
        elif isinstance(event, ToolCallEvent):
            call = event.tool_call
            if isinstance(call, AssistantFunctionToolCall):
                args = _preview(call.function.arguments, _TOOL_ARGS_PREVIEW)
                click.secho(f"\n[tool] {call.function.name}({args})", fg="yellow")
            else:
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            content = event.message.content
            if show_full:
                click.secho(f"[tool ok]\n{content}", fg="magenta")
            else:
                preview = _preview(content, _TOOL_RESULT_PREVIEW)
                # Single-line summary; full result stays in the message history for the model.
                one_line = preview.replace("\n", " ")
                click.secho(f"[tool ok] {one_line}", fg="magenta")
        elif isinstance(event, ErrorEvent):
            suffix = ""
            if event.source:
                suffix += f" source={event.source}"
            if not event.retryable:
                suffix += " (fatal)"
            click.secho(f"\n[error]{suffix} {event.message}", fg="bright_red")
        elif isinstance(event, CancelledEvent):
            if event.reason:
                click.secho(f"\n[cancelled] {event.reason}", fg="yellow")
            else:
                click.secho("\n[cancelled]", fg="yellow")
        elif isinstance(event, MaxRoundsEvent):
            if event.continued:
                click.secho(
                    f"\n[max rounds {event.rounds} reached — continuing with a higher allowance]",
                    fg="yellow",
                )
            else:
                click.secho(f"\n[max rounds reached: {event.rounds}]", fg="red")
        elif isinstance(event, UsageEvent):
            # Printed at end of turn as a summary; individual round usage is quiet.
            _ = event
        else:
            # AssistantMessageEvent — text already shown via TextDeltaEvent.
            _ = event
    if printed_text or printed_reasoning:
        click.echo()
    click.echo()
