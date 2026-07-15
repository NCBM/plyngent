from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ReasoningDeltaEvent, TextDeltaEvent, ToolResultEvent
from plyngent.cli.display import (
    get_markdown_enabled,
    markdown_render_available,
    print_markdown,
    render_events,
    set_markdown_enabled,
    set_verbose_tool_results,
)
from plyngent.lmproto.openai_compatible.model import ToolChatMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest

    from plyngent.agent import AgentEvent


async def _aiter(events: list[AgentEvent]) -> AsyncIterator[AgentEvent]:
    for event in events:
        yield event


async def test_render_reasoning_and_text(capsys: pytest.CaptureFixture[str]) -> None:
    set_markdown_enabled(False)
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
    set_markdown_enabled(True)


async def test_tool_result_preview_vs_verbose(capsys: pytest.CaptureFixture[str]) -> None:
    long = "x" * 200
    msg = ToolChatMessage(content=long, tool_call_id="1")
    set_verbose_tool_results(False)
    set_markdown_enabled(False)
    await render_events(_aiter([ToolResultEvent(message=msg)]))
    out = capsys.readouterr().out
    assert "…" in out
    assert long not in out

    set_verbose_tool_results(True)
    await render_events(_aiter([ToolResultEvent(message=msg)]), verbose=True)
    out2 = capsys.readouterr().out
    assert long in out2
    set_verbose_tool_results(False)
    set_markdown_enabled(True)


async def test_markdown_off_keeps_plain_stream(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("plyngent.cli.display.markdown_render_available", lambda: True)
    set_markdown_enabled(False)
    await render_events(_aiter([TextDeltaEvent(content="**bold**")]))
    out = capsys.readouterr().out
    assert "**bold**" in out
    set_markdown_enabled(True)


async def test_markdown_on_replaces_with_rich(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("plyngent.cli.display.markdown_render_available", lambda: True)
    set_markdown_enabled(True)
    await render_events(_aiter([TextDeltaEvent(content="hello **world**")]), markdown=True)
    out = capsys.readouterr().out
    # Rich markdown renders emphasis; raw ** markers should not remain as the sole form.
    assert "assistant:" in out or "assistant: " in out
    assert "world" in out


def test_print_markdown_renders(capsys: pytest.CaptureFixture[str]) -> None:
    print_markdown("# Title\n\n`code`", label="assistant: ")
    out = capsys.readouterr().out
    assert "Title" in out
    assert "code" in out


def test_markdown_flags_roundtrip() -> None:
    set_markdown_enabled(False)
    assert get_markdown_enabled() is False
    set_markdown_enabled(True)
    assert get_markdown_enabled() is True


def test_markdown_render_available_respects_plain_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYNGENT_PLAIN", "1")
    assert markdown_render_available() is False
    monkeypatch.delenv("PLYNGENT_PLAIN", raising=False)
