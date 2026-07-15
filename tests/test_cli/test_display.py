from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ReasoningDeltaEvent, TextDeltaEvent, ToolResultEvent
from plyngent.cli.display import render_events, set_verbose_tool_results
from plyngent.lmproto.openai_compatible.model import ToolChatMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest

    from plyngent.agent import AgentEvent


async def _aiter(events: list[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def test_render_reasoning_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                ReasoningDeltaEvent(content="think"),
                TextDeltaEvent(content="hello"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "reasoning: " in out
    assert "think" in out
    assert "assistant: " in out
    assert "hello" in out


async def test_tool_result_preview_vs_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    long = "x" * 200
    msg = ToolChatMessage(content=long, tool_call_id="1")
    set_verbose_tool_results(False)
    await render_events(_aiter([ToolResultEvent(message=msg)]))
    out = capsys.readouterr().out
    assert "…" in out
    assert long not in out

    set_verbose_tool_results(True)
    await render_events(_aiter([ToolResultEvent(message=msg)]), verbose=True)
    out2 = capsys.readouterr().out
    assert long in out2
    set_verbose_tool_results(False)
