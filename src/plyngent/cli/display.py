from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.agent import (
    CancelledEvent,
    ErrorEvent,
    MaxRoundsEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from plyngent.lmproto.openai_compatible.model import AssistantFunctionToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.agent import AgentEvent

_TOOL_RESULT_PREVIEW = 120
_TOOL_ARGS_PREVIEW = 80


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


async def render_events(events: AsyncIterator[AgentEvent]) -> None:  # noqa: C901, PLR0912
    """Print agent events to the terminal."""
    printed_text = False
    async for event in events:
        if isinstance(event, TextDeltaEvent):
            if not printed_text:
                click.echo()
                click.secho("assistant: ", fg="cyan", nl=False)
                printed_text = True
            click.echo(event.content, nl=False)
        elif isinstance(event, ToolCallEvent):
            call = event.tool_call
            if isinstance(call, AssistantFunctionToolCall):
                args = _preview(call.function.arguments, _TOOL_ARGS_PREVIEW)
                click.secho(f"\n[tool] {call.function.name}({args})", fg="yellow")
            else:
                click.secho(f"\n[tool] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            preview = _preview(event.message.content, _TOOL_RESULT_PREVIEW)
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
        else:
            # AssistantMessageEvent — text already shown via TextDeltaEvent.
            _ = event
    if printed_text:
        click.echo()
    click.echo()

