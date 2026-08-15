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
        function=AssistantFunctionTool(name="vcs_diff", arguments='{"path": ""}'),
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
            function=AssistantFunctionTool(name="vcs_diff", arguments='{"path": ""}'),
        )
    )
    await render_events(_aiter([call, _result("diff --git a/x b/x")]))
    out = capsys.readouterr().out
    assert "[tool] vcs_diff(" in out
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


async def test_pretty_verbose_shows_full_result(capsys: pytest.CaptureFixture[str]) -> None:
    """Verbose mode prints the full tool result even for pretty tools."""
    await render_events(
        _aiter([_read_call('{"path": "a.txt"}'), _result("L1-4\none\ntwo\n")]),
        verbose=True,
    )
    out = capsys.readouterr().out
    assert "[tool ok]" in out
    assert "L1-4\none\ntwo" in out
    assert "* Read" not in out


async def test_pretty_plain_output_has_no_ansi(capsys: pytest.CaptureFixture[str]) -> None:
    """Pretty summaries are plain text when stdout is not a TTY."""
    await render_events(_aiter([_read_call('{"path": "a.txt"}'), _result("L1-4\none\n")]))
    out = capsys.readouterr().out
    assert "\x1b[" not in out


async def test_tool_result_preview_first_line_and_count(capsys: pytest.CaptureFixture[str]) -> None:
    """Non-verbose preview shows the first line plus a line-count tail."""
    set_markdown_enabled(False)
    content = "first line\nsecond\nthird"
    await render_events(_aiter([ToolResultEvent(message=ToolChatMessage(content=content, tool_call_id="1"))]))
    out = capsys.readouterr().out
    assert "[tool ok] first line (…2 more lines)" in out

    await render_events(_aiter([ToolResultEvent(message=ToolChatMessage(content="only line", tool_call_id="2"))]))
    out = capsys.readouterr().out
    assert "[tool ok] only line" in out
    assert "more" not in out

    await render_events(_aiter([ToolResultEvent(message=ToolChatMessage(content="", tool_call_id="3"))]))
    out = capsys.readouterr().out
    assert "[tool ok] (no output)" in out
    set_markdown_enabled(True)


def _pretty_call(name: str, args_json: str) -> ToolCallEvent:
    return ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="1",
            function=AssistantFunctionTool(name=name, arguments=args_json),
        )
    )


async def test_pretty_listdir(capsys: pytest.CaptureFixture[str]) -> None:
    result = "dir\tsrc\nfile\tREADME.md\nfile\tpyproject.toml\n"
    await render_events(_aiter([_pretty_call("listdir", '{"path": "src"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* List 'src' (1 dirs, 2 files)" in out
    assert "[tool]" not in out
    assert "[tool ok]" not in out


async def test_pretty_listdir_empty(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("listdir", '{"path": "src"}'), _result("(empty)")]))
    out = capsys.readouterr().out
    assert "* List 'src' (empty)" in out


async def test_pretty_listdir_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("listdir", '{"path": "nope"}'), _result("error: not a directory: nope")]))
    out = capsys.readouterr().out
    assert "* List 'nope' (error: not a directory: nope)" in out


async def test_pretty_glob(capsys: pytest.CaptureFixture[str]) -> None:
    result = "src/a.py\nsrc/b.py\n"
    await render_events(_aiter([_pretty_call("glob_paths", '{"pattern": "**/*.py"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Glob '**/*.py' in '.' (2 paths)" in out


async def test_pretty_glob_no_matches(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("glob_paths", '{"pattern": "**/*.rs"}'), _result("(no matches)")]))
    out = capsys.readouterr().out
    assert "* Glob '**/*.rs' in '.' (no matches)" in out


async def test_pretty_glob_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    result = "a\nb\n...[truncated at 200 matches]"
    await render_events(_aiter([_pretty_call("glob_paths", '{"pattern": "*"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Glob '*' in '.' (2 paths)" in out


async def test_pretty_grep(capsys: pytest.CaptureFixture[str]) -> None:
    result = "src/a.py:3: x = 1\nsrc/a.py:9: x = 2\nsrc/b.py:1: y = 3\n"
    await render_events(_aiter([_pretty_call("grep_files", '{"pattern": "x"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Grep 'x' in '.' (3 matches in 2 files)" in out


async def test_pretty_grep_no_matches(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("grep_files", '{"pattern": "zzz"}'), _result("(no matches)")]))
    out = capsys.readouterr().out
    assert "* Grep 'zzz' in '.' (no matches)" in out


async def test_pretty_run_argv_success(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "exit_code=0\ntimed_out=false\ncwd=.\ncmd=git status --short\n--- stdout ---\n M src/foo.py\n--- stderr ---\n"
    )
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["git", "status", "--short"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run 'git status --short' → exit 0 (1 line)" in out
    assert "[tool]" not in out
    assert "[tool ok]" not in out


async def test_pretty_run_argv_failure(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "exit_code=1\ntimed_out=false\ncwd=.\ncmd=ls /nope\n--- stdout ---\n--- stderr ---\nls: cannot access '/nope'\n"
    )
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["ls", "/nope"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run 'ls /nope' → exit 1 (1 line)" in out


async def test_pretty_run_argv_timed_out(capsys: pytest.CaptureFixture[str]) -> None:
    result = "exit_code=\ntimed_out=true\ncwd=.\ncmd=sleep 10\n--- stdout ---\n--- stderr ---\n"
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["sleep", "10"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run 'sleep 10' → timed out (0 lines)" in out


async def test_pretty_run_argv_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter([_pretty_call("run_argv", '{"argv": ["nope"]}'), _result("error: executable not found: 'nope'")])
    )
    out = capsys.readouterr().out
    assert "* Run 'nope' (error: executable not found: 'nope')" in out


async def test_pretty_run_argv_batch_stopped(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "steps=3 ran=2 stop_on_error=true stopped_early=true\n"
        "--- step 1 ---\nexit_code=1\ntimed_out=false\ncwd=.\ncmd=git status\n"
        "pipe_out=false\nmix_stderr=false\n--- stdout ---\n--- stderr ---\n"
        "--- step 2 ---\nexit_code=0\ntimed_out=false\ncwd=.\ncmd=git diff\n"
        "pipe_out=false\nmix_stderr=false\n--- stdout ---\n--- stderr ---\n"
        "--- summary ---\nlast_exit=1\n"
    )
    await render_events(_aiter([_pretty_call("run_argv_batch", '{"steps": []}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Batch (2/3 steps ran, stopped early)" in out


async def test_pretty_run_argv_batch_complete(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "steps=2 ran=2 stop_on_error=true stopped_early=false\n"
        "--- step 1 ---\nexit_code=0\n--- stdout ---\n--- stderr ---\n"
        "--- step 2 ---\nexit_code=0\n--- stdout ---\n--- stderr ---\n"
        "--- summary ---\nlast_exit=0\n"
    )
    await render_events(_aiter([_pretty_call("run_argv_batch", '{"steps": []}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Batch (2/2 steps ran, last exit 0)" in out


async def test_pretty_fetch(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=200\nmethod=GET\nfinal_url=https://example.com\ncontent_type=application/json\n"
        "body_kind=json\nbytes=1229\ntruncated=false\nredirects=0\nsecurity=public\n--- body ---\n{...}\n"
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* GET 200 https://example.com (json, 1.2KB)" in out
    assert "[tool]" not in out


async def test_pretty_fetch_error_status(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=404\nmethod=GET\nfinal_url=https://example.com/nope\ncontent_type=text/html\n"
        "body_kind=html\nbytes=45\ntruncated=false\nredirects=0\nsecurity=public\n--- body ---\nNot found\n"
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com/nope"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* GET 404 https://example.com/nope (html, 45B)" in out


async def test_pretty_fetch_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=200\nmethod=GET\nfinal_url=https://example.com\ncontent_type=text/plain\n"
        "body_kind=text\nbytes=50000\ntruncated=true\nredirects=0\nsecurity=public\n--- body ---\n..."
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* GET 200 https://example.com (text, 48.8KB, truncated)" in out


async def test_pretty_fetch_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result("error: fetch failed: boom")])
    )
    out = capsys.readouterr().out
    assert "* GET (error: fetch failed: boom)" in out
