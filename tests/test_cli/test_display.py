from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ReasoningDeltaEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from plyngent.cli.display import (
    get_markdown_enabled,
    markdown_render_available,
    print_markdown,
    render_events,
    set_markdown_enabled,
    set_verbose_tool_results,
)
from plyngent.lmproto.openai_compatible.model import (
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    ToolChatMessage,
)

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
    assert "reasoning:" in out
    assert "think" in out
    assert "assistant:" in out
    assert "hello" in out
    # Labels on their own lines (content begins after newline).
    assert "assistant:\nhello" in out or "assistant:\r\nhello" in out
    set_markdown_enabled(True)


async def test_flush_markdown_on_source_change(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assistant segment before tools is flushed so later text is a new segment."""
    monkeypatch.setattr("plyngent.cli.display.markdown_render_available", lambda: True)
    set_markdown_enabled(True)
    from plyngent.agent import ToolCallEvent
    from plyngent.lmproto.openai_compatible.model import (
        AssistantFunctionTool,
        AssistantFunctionToolCall,
    )

    call = AssistantFunctionToolCall(
        id="1",
        function=AssistantFunctionTool(name="run_argv", arguments='{"argv": ["echo", "x"]}'),
    )
    await render_events(
        _aiter(
            [
                TextDeltaEvent(content="before **tool**"),
                ToolCallEvent(tool_call=call),
                TextDeltaEvent(content="after"),
            ]
        ),
        markdown=True,
    )
    out = capsys.readouterr().out
    assert "[tool]" in out
    assert "after" in out


def _read_call(args_json: str) -> ToolCallEvent:
    return ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="1",
            function=AssistantFunctionTool(name="read_file", arguments=args_json),
        )
    )


def _result(content: str) -> ToolResultEvent:
    return ToolResultEvent(message=ToolChatMessage(content=content, tool_call_id="1"))


async def test_pretty_read_file_done(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_read_call('{"path": "a.txt"}'), _result("L1-4\none\ntwo\n")]))
    out = capsys.readouterr().out
    assert "* Read 'a.txt' L1-4 (done)" in out
    assert "[tool]" not in out
    assert "[tool ok]" not in out


async def test_pretty_read_file_statuses(capsys: pytest.CaptureFixture[str]) -> None:
    cases = [
        ("error: file not found: a.txt", "(file not found)"),
        ("error: not a file: a.txt", "(not a file)"),
        ("error: something else", "(error)"),
        ("no header content", "(done)"),
    ]
    for content, status in cases:
        await render_events(_aiter([_read_call('{"path": "a.txt"}'), _result(content)]))
        out = capsys.readouterr().out
        assert f"* Read 'a.txt' {status}" in out, f"status {status} not rendered for {content!r}"


def _tree_call(args_json: str) -> ToolCallEvent:
    return ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="1",
            function=AssistantFunctionTool(name="tree", arguments=args_json),
        )
    )


async def test_pretty_tree_markdown(capsys: pytest.CaptureFixture[str]) -> None:
    result = "- src/\n  - main.py\n  - nested/\n    - deep.txt\n- a.txt\n"
    await render_events(_aiter([_tree_call('{"path": "."}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Tree '.' (2 dirs, 3 files, depth 3)" in out
    assert "[tool]" not in out


async def test_pretty_tree_flat_omits_depth(capsys: pytest.CaptureFixture[str]) -> None:
    result = "src/\nsrc/main.py\na.txt\n… (5 more paths not shown)"
    await render_events(_aiter([_tree_call('{"path": ".", "format": "flat"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Tree '.' (1 dirs, 2 files)" in out
    assert "depth" not in out


async def test_pretty_tree_decorated(capsys: pytest.CaptureFixture[str]) -> None:
    result = "src/\n├── main.py\n└── nested/\n    └── deep.txt\n"
    await render_events(_aiter([_tree_call('{"path": "src", "format": "decorated"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Tree 'src' (1 dirs, 2 files, depth 2)" in out


async def test_pretty_tree_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_tree_call('{"path": "nope"}'), _result("error: not a directory: nope")]))
    out = capsys.readouterr().out
    assert "* Tree 'nope' (error)" in out


async def test_pretty_todo_push(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="2",
            function=AssistantFunctionTool(name="todo_push", arguments='{"titles": ["T1"]}'),
        )
    )
    result = (
        "pushed group (depth=1) items=[a1]\n"
        "(LIFO of groups: depth=1; TOP group = current breakdown level)\n"
        "group d=0 TOP:\n"
        "  [ ] a1: T1"
    )
    await render_events(_aiter([call, _result(result)]))
    out = capsys.readouterr().out
    assert "* Todo Push:" in out
    assert "  [ ] a1: T1" in out
    assert "[tool]" not in out


async def test_non_pretty_tool_keeps_old_style(capsys: pytest.CaptureFixture[str]) -> None:
    call = ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="3",
            function=AssistantFunctionTool(name="run_argv", arguments='{"argv": ["ls"]}'),
        )
    )
    await render_events(_aiter([call, _result("exit_code=0")]))
    out = capsys.readouterr().out
    assert "[tool] run_argv(" in out
    assert "[tool ok]" in out


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
