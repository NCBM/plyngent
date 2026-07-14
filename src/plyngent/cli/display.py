from __future__ import annotations

from typing import TYPE_CHECKING

import click

from plyngent.agent import (
    MaxRoundsEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from plyngent.lmproto.openai_compatible.model import AssistantFunctionToolCall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from plyngent.agent import AgentEvent

_TOOL_RESULT_PREVIEW = 200


async def render_events(events: AsyncIterator[AgentEvent]) -> None:
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
                click.secho(
                    f"\n[tool call] {call.function.name}({call.function.arguments})",
                    fg="yellow",
                )
            else:
                click.secho(f"\n[tool call] custom id={call.id}", fg="yellow")
        elif isinstance(event, ToolResultEvent):
            content = event.message.content
            preview = (
                content
                if len(content) <= _TOOL_RESULT_PREVIEW
                else content[:_TOOL_RESULT_PREVIEW] + "…"
            )
            click.secho(f"[tool result] {preview}", fg="magenta")
        elif isinstance(event, MaxRoundsEvent):
            click.secho(f"\n[max rounds reached: {event.rounds}]", fg="red")
        else:
            # AssistantMessageEvent — text already shown via TextDeltaEvent.
            _ = event
    if printed_text:
        click.echo()
