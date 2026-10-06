from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ErrorEvent, ReasoningDeltaEvent, TextDeltaEvent, ToolCallEvent, ToolResultEvent
from plyngent.cli.display import (
    _clear_streamed_lines,
    _line_count_for_clear,
    _pretty_line_for,
    close_open_pretty_line,
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
        function=AssistantFunctionTool(name="acme_ping", arguments='{"host": "example.com"}'),
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


async def test_pretty_read_file_numbered_no_range(capsys: pytest.CaptureFixture[str]) -> None:
    """Numbered reads carry no L header, so the summary falls back to ``(done)``."""
    content = "    1|one\n    2|two\n"
    await render_events(_aiter([_read_call('{"path": "a.txt", "with_lineno": true}'), _result(content)]))
    out = capsys.readouterr().out
    assert "* Read 'a.txt' (done)" in out


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


async def test_pretty_todo_batch_merges_to_one_stack(capsys: pytest.CaptureFixture[str]) -> None:
    """A batch of todo calls: one summary line, then the last result's stack."""
    stack = "(LIFO of groups: depth=1; TOP group = current breakdown level)\ngroup d=0 TOP:\n  [ ] a1: T1"
    await render_events(
        _aiter(
            [
                _pretty_call("todo_push", '{"titles": ["T1"]}'),
                _pretty_call("todo_update", '{"item_id": "a1", "status": "done"}'),
                _result(f"pushed group (depth=1) items=[a1]\n{stack}"),
                _result(f"updated a1 → done: T1\n{stack}"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Todo x2: Push, Update" in out
    assert "* Todo Push:" not in out
    assert "* Todo Update:" not in out
    # Only the last result is printed, so the stack renders once.
    assert out.count("group d=0 TOP:") == 1
    assert "pushed group" not in out
    assert "updated a1 → done: T1" in out


async def test_pretty_todo_batch_mixed_with_other_tool_stays_per_call(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch holding a non-todo call keeps one line (and stack) per todo call."""
    stack = "group d=0 TOP:\n  [ ] a1: T1"
    await render_events(
        _aiter(
            [
                _pretty_call("todo_push", '{"titles": ["T1"]}'),
                _pretty_call("listdir", '{"path": "src"}'),
                _result(f"pushed group (depth=1) items=[a1]\n{stack}"),
                _result("dir\tsrc\nfile\tREADME.md\n"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Todo Push:" in out
    assert "group d=0 TOP:" in out
    assert "src" in out
    assert "* Todo x" not in out


async def test_non_pretty_tool_keeps_old_style(capsys: pytest.CaptureFixture[str]) -> None:
    """Tools with no pretty line (plugins, unknown names) keep the ``[tool]`` style."""
    call = ToolCallEvent(
        tool_call=AssistantFunctionToolCall(
            id="3",
            function=AssistantFunctionTool(name="acme_ping", arguments='{"host": "example.com"}'),
        )
    )
    await render_events(_aiter([call, _result("pong")]))
    out = capsys.readouterr().out
    assert "[tool] acme_ping(" in out
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


async def test_pretty_regex_files(capsys: pytest.CaptureFixture[str]) -> None:
    result = "src/a.py:3: x = 1\nsrc/a.py:9: x = 2\nsrc/b.py:1: y = 3\n"
    await render_events(_aiter([_pretty_call("regex_files", '{"pattern": "x"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Search 'x' in '.' (3 matches in 2 files)" in out


async def test_pretty_regex_files_no_matches(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("regex_files", '{"pattern": "zzz"}'), _result("(no matches)")]))
    out = capsys.readouterr().out
    assert "* Search 'zzz' in '.' (no matches)" in out


async def test_pretty_run_argv_success(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "exit_code=0\ntimed_out=false\ncwd=.\ncmd=git status --short\n--- stdout ---\n M src/foo.py\n--- stderr ---\n"
    )
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["git", "status", "--short"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run $ git status --short (exit code 0)" in out
    assert "[tool]" not in out
    assert "[tool ok]" not in out


async def test_pretty_run_argv_failure(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "exit_code=1\ntimed_out=false\ncwd=.\ncmd=ls /nope\n--- stdout ---\n--- stderr ---\nls: cannot access '/nope'\n"
    )
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["ls", "/nope"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run $ ls /nope (exit code 1)" in out


async def test_pretty_run_argv_timed_out(capsys: pytest.CaptureFixture[str]) -> None:
    result = "exit_code=\ntimed_out=true\ncwd=.\ncmd=sleep 10\n--- stdout ---\n--- stderr ---\n"
    await render_events(_aiter([_pretty_call("run_argv", '{"argv": ["sleep", "10"]}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run $ sleep 10 (timed out)" in out


async def test_pretty_run_argv_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter([_pretty_call("run_argv", '{"argv": ["nope"]}'), _result("error: executable not found: 'nope'")])
    )
    out = capsys.readouterr().out
    assert "* Run $ nope (error: executable not found: 'nope')" in out


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
    assert "* Run batch (stopped early)" in out


async def test_pretty_run_argv_batch_complete(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "steps=2 ran=2 stop_on_error=true stopped_early=false\n"
        "--- step 1 ---\nexit_code=0\n--- stdout ---\n--- stderr ---\n"
        "--- step 2 ---\nexit_code=0\n--- stdout ---\n--- stderr ---\n"
        "--- summary ---\nlast_exit=0\n"
    )
    await render_events(_aiter([_pretty_call("run_argv_batch", '{"steps": []}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Run batch (done)" in out


async def test_pretty_fetch(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=200\nmethod=GET\nfinal_url=https://example.com\ncontent_type=application/json\n"
        "body_kind=json\nbytes=1229\ntruncated=false\nredirects=0\nsecurity=public\n--- body ---\n{...}\n"
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Fetch GET https://example.com (200)" in out
    assert "[tool]" not in out


async def test_pretty_fetch_error_status(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=404\nmethod=GET\nfinal_url=https://example.com/nope\ncontent_type=text/html\n"
        "body_kind=html\nbytes=45\ntruncated=false\nredirects=0\nsecurity=public\n--- body ---\nNot found\n"
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com/nope"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Fetch GET https://example.com/nope (404)" in out


async def test_pretty_fetch_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "status=200\nmethod=GET\nfinal_url=https://example.com\ncontent_type=text/plain\n"
        "body_kind=text\nbytes=50000\ntruncated=true\nredirects=0\nsecurity=public\n--- body ---\n..."
    )
    await render_events(_aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* Fetch GET https://example.com (200)" in out


async def test_pretty_fetch_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter([_pretty_call("fetch", '{"url": "https://example.com"}'), _result("error: fetch failed: boom")])
    )
    out = capsys.readouterr().out
    assert "* Fetch GET https://example.com (error: fetch failed: boom)" in out


async def test_pretty_mutators(capsys: pytest.CaptureFixture[str]) -> None:
    cases = [
        ("write_file", '{"path": "src/x.py"}', "wrote 120 characters to src/x.py", "* Write 'src/x.py' (120 chars)"),
        (
            "edit_replace",
            '{"path": "src/a.py"}',
            "replaced 1 occurrence in src/a.py ('x' → 'y')",
            "* Edit 'src/a.py' (done)",
        ),
        (
            "edit_lineno",
            '{"path": "src/a.py"}',
            "replaced lines 3-5 (3 lines; first: 'pass') with 2 lines in src/a.py",
            "* Edit 'src/a.py' (done)",
        ),
        (
            "copy_path",
            '{"src": "a.txt", "dst": "b.txt"}',
            "copied file a.txt -> b.txt",
            "* Copy 'a.txt' → 'b.txt' (done)",
        ),
        ("move_path", '{"src": "src", "dst": "dst"}', "moved directory src -> dst", "* Move 'src' → 'dst' (done)"),
        ("delete_path", '{"path": "tmp/x.txt"}', "deleted file tmp/x.txt", "* Delete 'tmp/x.txt' (done)"),
    ]
    for name, args, content, expected in cases:
        await render_events(_aiter([_pretty_call(name, args), _result(content)]))
        out = capsys.readouterr().out
        assert expected in out, f"{name}: expected {expected!r} in {out!r}"
        assert "[tool ok]" not in out


async def test_pretty_mutator_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter([_pretty_call("copy_path", '{"src": "x", "dst": "y"}'), _result("error: source does not exist: x")])
    )
    out = capsys.readouterr().out
    assert "* Copy 'x' → 'y' (error: source does not exist: x)" in out


async def test_pretty_vcs_status_done(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("vcs_status", "{}"), _result("(clean)")]))
    out = capsys.readouterr().out
    assert "* VCS Status (done)" in out
    assert "[tool]" not in out


async def test_pretty_vcs_status_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                _pretty_call("vcs_status", "{}"),
                _result("error: no supported VCS detected under workspace (currently: git)"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* VCS Status (error: no supported VCS detected under workspace (currently: git))" in out


async def test_pretty_vcs_log_count(capsys: pytest.CaptureFixture[str]) -> None:
    result = (
        "5436457 2026-08-15 worldmozara core/tools: allow edit_lineno append\nabc1234 2026-08-14 worldmozara cli: fix\n"
    )
    await render_events(_aiter([_pretty_call("vcs_log", '{"limit": 2}'), _result(result)]))
    out = capsys.readouterr().out
    assert "* VCS Log (2 commits)" in out


async def test_pretty_vcs_log_single(capsys: pytest.CaptureFixture[str]) -> None:
    content = "abc1234 2026-08-14 worldmozara x\n"
    await render_events(_aiter([_pretty_call("vcs_log", '{"limit": 1}'), _result(content)]))
    out = capsys.readouterr().out
    assert "* VCS Log (1 commit)" in out


async def test_pretty_vcs_log_no_commits(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("vcs_log", "{}"), _result("(no commits)")]))
    out = capsys.readouterr().out
    assert "* VCS Log (no commits)" in out


async def test_pretty_wait_waited(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("wait", '{"duration": 5}'), _result("waited 5s")]))
    out = capsys.readouterr().out
    assert "* Wait (5s)" in out
    assert "[tool]" not in out


async def test_pretty_wait_disturbed(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("wait", '{"duration": 30}'), _result("disturbed by user: ship it")]))
    out = capsys.readouterr().out
    assert "* Wait (disturbed)" in out


async def test_pretty_wait_cancelled(capsys: pytest.CaptureFixture[str]) -> None:
    """Ctrl+C during the wait prompt reads as cancelled, not disturbed."""
    await render_events(_aiter([_pretty_call("wait", '{"duration": 30}'), _result("cancelled by user")]))
    out = capsys.readouterr().out
    assert "* Wait (cancelled)" in out


async def test_pretty_wait_error(capsys: pytest.CaptureFixture[str]) -> None:
    content = "error: duration must not be negative"
    await render_events(_aiter([_pretty_call("wait", '{"duration": -1}'), _result(content)]))
    out = capsys.readouterr().out
    assert "* Wait (error: duration must not be negative)" in out


async def test_pretty_pty_open_and_read(capsys: pytest.CaptureFixture[str]) -> None:
    opened = "session_id=3\nalive=true\nexit_code=\ncmd=ls -la"
    await render_events(_aiter([_pretty_call("open_pty", '{"command": ["ls", "-la"]}'), _result(opened)]))
    out = capsys.readouterr().out
    assert "* PTY Open $ ls -la (session 3)" in out

    read = (
        "session_id=3\nalive=true\nexit_code=\nmatched=false\ntruncated=false\n"
        "budget_exhausted=false\n--- data ---\nhello\n"
    )
    await render_events(_aiter([_pretty_call("read_pty", '{"session_id": 3}'), _result(read)]))
    out = capsys.readouterr().out
    assert "* PTY Read 3 (alive, 1 line)" in out


async def test_pretty_pty_read_exited_and_matched(capsys: pytest.CaptureFixture[str]) -> None:
    """An exited session reports the code; ``until`` reports that it matched."""
    exited = (
        "session_id=3\nalive=false\nexit_code=0\nmatched=false\ntruncated=false\n"
        "budget_exhausted=false\n--- data ---\nbye\n"
    )
    await render_events(_aiter([_pretty_call("read_pty", '{"session_id": 3}'), _result(exited)]))
    out = capsys.readouterr().out
    assert "* PTY Read 3 (exited 0)" in out

    matched = (
        "session_id=3\nalive=true\nexit_code=\nmatched=true\ntruncated=false\n"
        "budget_exhausted=false\n--- data ---\nprompt\n$ \n"
    )
    await render_events(_aiter([_pretty_call("read_pty", '{"session_id": 3, "until": "$ "}'), _result(matched)]))
    out = capsys.readouterr().out
    assert "* PTY Read 3 (matched, 2 lines)" in out


async def test_pretty_pty_read_empty_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    empty = (
        "session_id=3\nalive=true\nexit_code=\nmatched=false\ntruncated=false\nbudget_exhausted=false\n--- data ---\n"
    )
    await render_events(_aiter([_pretty_call("read_pty", '{"session_id": 3}'), _result(empty)]))
    out = capsys.readouterr().out
    assert "* PTY Read 3 (alive, no output)" in out

    await render_events(
        _aiter([_pretty_call("read_pty", '{"session_id": 9}'), _result("error: unknown PTY session 9")])
    )
    out = capsys.readouterr().out
    assert "* PTY Read 9 (error: unknown PTY session 9)" in out


async def test_pretty_pty_write_and_keys(capsys: pytest.CaptureFixture[str]) -> None:
    status = "session_id=3\nalive=true\nexit_code=\nwrote=5"
    await render_events(_aiter([_pretty_call("write_pty", '{"session_id": 3, "data": "ls -la"}'), _result(status)]))
    out = capsys.readouterr().out
    assert "* PTY Write 3 (5 chars)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("write_pty_keys", '{"session_id": 3, "data": "ctrl+c"}'),
                _result("session_id=3\nalive=true\nexit_code=\nwrote=1"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* PTY Keys 3 (1 char)" in out


async def test_pretty_ask_into_pty_never_shows_the_answer(capsys: pytest.CaptureFixture[str]) -> None:
    """The result only says the answer was written; the payload stays local."""
    status = "session_id=3\nalive=true\nexit_code=\nwrote=true\nsecret=true\nsubmit=true\nsource=human"
    await render_events(
        _aiter([_pretty_call("ask_into_pty", '{"session_id": 3, "message": "sudo password"}'), _result(status)])
    )
    out = capsys.readouterr().out
    assert "* PTY Ask 3 (answer sent, secret)" in out
    assert "wrote=true" not in out


async def test_pretty_close_pty(capsys: pytest.CaptureFixture[str]) -> None:
    closed = "session_id=3\nclosed=true\nalive=false\nexit_code=0\nmessage=closed"
    await render_events(_aiter([_pretty_call("close_pty", '{"session_id": 3}'), _result(closed)]))
    out = capsys.readouterr().out
    assert "* PTY Close 3 (closed)" in out

    already = "session_id=3\nclosed=false\nalive=false\nexit_code=0\nmessage=already closed"
    await render_events(_aiter([_pretty_call("close_pty", '{"session_id": 3}'), _result(already)]))
    out = capsys.readouterr().out
    assert "* PTY Close 3 (already closed)" in out


async def test_pretty_ask_user_line_and_choice(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("ask_user_line", '{"question": "Which port?"}'), _result("8080")]))
    out = capsys.readouterr().out
    assert "* Ask 'Which port?' (answered)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("ask_user_choice", '{"question": "Pick one", "options": ["alpha", "beta"]}'),
                _result("beta"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Ask Choose 'Pick one' (beta)" in out


async def test_pretty_ask_user_form_counts_answers(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                _pretty_call("ask_user_form", '{"title": "Setup", "fields": [{"name": "port", "prompt": "Port?"}]}'),
                _result('{"port": 8080, "host": "localhost"}'),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Ask Form 'Setup' (2 answers)" in out


async def test_pretty_ask_user_error(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                _pretty_call("ask_user_line", '{"question": "Port?"}'),
                _result("error: interactive prompts are unavailable (stdin is not a TTY)"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Ask 'Port?' (error: interactive prompts are unavailable (stdin is not a TTY))" in out


async def test_pretty_request_directory_access(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                _pretty_call("request_directory_access", '{"path": "/data", "mode": "write"}'),
                _result("access granted: /data (write, session)\nnote: the path denylist still applies"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Access '/data' (write) (granted)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("request_directory_access", '{"path": "/data"}'),
                _result("already accessible: /data is inside the workspace"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Access '/data' (read) (already accessible)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("request_directory_access", '{"path": "/root"}'),
                _result("error: access to /root denied (declined or timed out after 30s)"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "(error: access to /root denied (declined or timed out after 30s))" in out


async def test_pretty_new_temporary_workspace(capsys: pytest.CaptureFixture[str]) -> None:
    content = "temporary_workspace=/tmp/plyngent-ws-abc\nnote: project workspace unchanged"
    await render_events(_aiter([_pretty_call("new_temporary_workspace", '{"prefix": "ws"}'), _result(content)]))
    out = capsys.readouterr().out
    assert "* Temp Dir (/tmp/plyngent-ws-abc)" in out


async def test_pretty_get_truncated(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("get_truncated", '{"token": "abc"}'), _result("L81-160\nmore\n")]))
    out = capsys.readouterr().out
    assert "* Resume Truncated L81-160 (done)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("get_truncated", '{"token": "abc"}'),
                _result("error: truncate token expired (no longer in memory); re-run the tool"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "(error: truncate token expired (no longer in memory); re-run the tool)" in out


async def test_pretty_vcs_kind_branch_and_diff(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(_aiter([_pretty_call("vcs_kind", "{}"), _result("git")]))
    out = capsys.readouterr().out
    assert "* VCS Kind (git)" in out

    await render_events(_aiter([_pretty_call("vcs_branch", "{}"), _result("main")]))
    out = capsys.readouterr().out
    assert "* VCS Branch (main)" in out

    diff = (
        "diff --git a/x.py b/x.py\nindex 111..222 100644\n--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n-old\n+new\n+extra\n"
    )
    await render_events(_aiter([_pretty_call("vcs_diff", '{"path": ""}'), _result(diff)]))
    out = capsys.readouterr().out
    assert "* VCS Diff (1 file, +2/-1)" in out

    await render_events(_aiter([_pretty_call("vcs_diff", '{"staged": true}'), _result("(no diff)")]))
    out = capsys.readouterr().out
    assert "* VCS Diff (staged) (no diff)" in out


async def test_pretty_todo_list_pop_and_clear(capsys: pytest.CaptureFixture[str]) -> None:
    stack = "  [ ] a1: T1\n  [x] a2: T2"
    await render_events(_aiter([_pretty_call("todo_list", "{}"), _result(stack)]))
    out = capsys.readouterr().out
    assert "* Todo List:" in out
    assert "  [ ] a1: T1" in out

    await render_events(_aiter([_pretty_call("todo_pop", "{}"), _result(f"popped TOP group (a1:T1)\n{stack}")]))
    out = capsys.readouterr().out
    assert "* Todo Pop:" in out

    await render_events(_aiter([_pretty_call("todo_clear", "{}"), _result("cleared 3 item(s)")]))
    out = capsys.readouterr().out
    assert "* Todo Clear:" in out
    assert "cleared 3 item(s)" in out


async def test_pretty_mcp_tools_use_one_generic_line(capsys: pytest.CaptureFixture[str]) -> None:
    """Namespaced MCP tools render as ``* MCP <server>:<tool> ``."""
    await render_events(_aiter([_pretty_call("mcp__docs__search", '{"query": "x"}'), _result("found 1 page")]))
    out = capsys.readouterr().out
    assert "* MCP docs:search (done)" in out
    assert "[tool]" not in out

    await render_events(_aiter([_pretty_call("mcp__docs__search", '{"query": "x"}'), _result("error: server died")]))
    out = capsys.readouterr().out
    assert "* MCP docs:search (error: server died)" in out


def test_every_builtin_tool_has_a_pretty_line() -> None:
    """A new builtin tool must ship a renderer, or it silently loses the syntax."""
    from plyngent.tools.catalog import default_tool_definitions

    missing = [d.name for d in default_tool_definitions() if _pretty_line_for(d.name) is None]
    assert missing == []


async def test_pretty_skill_tools(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [_pretty_call("skill_list", "{}"), _result("skills: 3\npdf-processing  [plyngent-user]  /x — Extract PDFs")]
        )
    )
    out = capsys.readouterr().out
    assert "* Skills (3 skills)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("skill_read", '{"name": "pdf-processing"}'),
                _result("L1-40\n# PDF processing\n"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill 'pdf-processing' L1-40 (done)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("skill_read", '{"name": "pdf-processing", "file": "scripts/run.sh"}'),
                _result("L1-3\n#!/bin/sh\n"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill 'pdf-processing' scripts/run.sh L1-3 (done)" in out

    await render_events(
        _aiter(
            [
                _pretty_call("skill_search", '{"pattern": "pdftotext"}'),
                _result("alpha/SKILL.md:3: Use pdftotext.\nbeta/notes.md:1: pdftotext again"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill Search 'pdftotext' (2 matches in 2 skills)" in out

    await render_events(_aiter([_pretty_call("skill_search", '{"pattern": "nope"}'), _result("(no matches)")]))
    out = capsys.readouterr().out
    assert "* Skill Search 'nope' (no matches)" in out


async def test_pretty_skill_create_and_edit(capsys: pytest.CaptureFixture[str]) -> None:
    await render_events(
        _aiter(
            [
                _pretty_call("skill_create", '{"name": "acme", "description": "d", "body": "b"}'),
                _result("created skill 'acme' at /skills/acme\nnotes: SKILL.md has no frontmatter description"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill Create 'acme' (created) — SKILL.md has no frontmatter description" in out

    await render_events(
        _aiter(
            [
                _pretty_call("skill_edit", '{"name": "acme"}'),
                _result("updated skill 'acme' (SKILL.md: replaced 1 of 1 occurrence(s))"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill Edit 'acme' (SKILL.md: replaced 1 of 1 occurrence(s))" in out

    await render_events(
        _aiter(
            [
                _pretty_call("skill_edit", '{"name": "acme", "file": "scripts/run.sh"}'),
                _result("error: skill write to /skills/acme denied (declined or timed out after 30s)"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Skill Edit 'acme' scripts/run.sh (error: skill write to /skills/acme denied" in out


async def test_pretty_prefix_shown_while_call_runs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An interactive terminal shows the prefix before the tool has a result."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    while_running: list[str] = []

    async def events() -> AsyncIterator[AgentEvent]:
        yield _read_call('{"path": "a.txt"}')
        while_running.append(capsys.readouterr().out)  # tool still running here
        yield _result("L1-4\none\ntwo\n")

    await render_events(events())
    assert while_running == ["\n* Read 'a.txt' "]
    assert capsys.readouterr().out == "L1-4 (done)\n\n"


async def test_pretty_prefix_absent_when_not_a_tty(capsys: pytest.CaptureFixture[str]) -> None:
    """Non-interactive output stays one whole-line write (pipes, logs, tests)."""
    while_running: list[str] = []

    async def events() -> AsyncIterator[AgentEvent]:
        yield _read_call('{"path": "a.txt"}')
        while_running.append(capsys.readouterr().out)
        yield _result("L1-4\none\ntwo\n")

    await render_events(events())
    assert while_running == [""]
    assert capsys.readouterr().out == "\n* Read 'a.txt' L1-4 (done)\n\n"


async def test_pretty_verbose_keeps_full_result_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verbose mode stays a full log: no prefix line, whole result."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    set_verbose_tool_results(True)
    try:
        await render_events(_aiter([_read_call('{"path": "a.txt"}'), _result("L1-4\none\n")]))
    finally:
        set_verbose_tool_results(False)
    out = capsys.readouterr().out
    assert "[tool ok]\nL1-4\none\n" in out
    assert "* Read 'a.txt' " not in out


async def test_pretty_parallel_batch_gives_every_call_its_own_row(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A second call in the batch holds its prefix until the first row is done."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    await render_events(
        _aiter(
            [
                _pretty_call("read_file", '{"path": "a.txt"}'),
                _pretty_call("read_file", '{"path": "b.txt"}'),
                _result("L1-4\none\n"),
                _result("L1-9\ntwo\n"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert out == ("\n* Read 'a.txt' L1-4 (done)\n\n* Read 'b.txt' L1-9 (done)\n\n")
    assert "\x1b[" not in out  # the cursor is never moved


async def test_pretty_edit_batch_prints_one_row_per_edit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two edits in one round: one finished row each, no abandoned prefix row."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    await render_events(
        _aiter(
            [
                _pretty_call("edit_replace", '{"path": "a.py"}'),
                _pretty_call("edit_replace", '{"path": "b.py"}'),
                _result("replaced 1 occurrence in a.py ('x' → 'y')"),
                _result("replaced 1 occurrence in b.py ('x' → 'y')"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert out == "\n* Edit 'a.py' (done)\n\n* Edit 'b.py' (done)\n\n"


async def test_pretty_parallel_batch_keeps_the_line_above_the_prefix(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rendering a batch must not touch the line above the prefix.

    Reasoning streams without a trailing newline, so the prefix line sits right
    below the reasoning text; moving the cursor up used to wipe that reasoning
    line out of the screen.
    """
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    await render_events(
        _aiter(
            [
                ReasoningDeltaEvent(content="think"),
                _pretty_call("read_file", '{"path": "a.txt"}'),
                _pretty_call("read_file", '{"path": "b.txt"}'),
                _result("L1-4\none\n"),
                _result("L1-9\ntwo\n"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert out == ("\nreasoning:\nthink\n* Read 'a.txt' L1-4 (done)\n\n* Read 'b.txt' L1-9 (done)\n\n\n")
    assert "\x1b[" not in out


async def test_pretty_line_closed_by_out_of_band_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A prompt/confirm printed while the call runs ends the open prefix line.

    The prompt backend runs :func:`close_open_pretty_line` before writing, so its
    box starts on a fresh line and the result then prints whole.
    """
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)

    async def _events():
        yield _pretty_call("read_file", '{"path": "a.txt"}')
        close_open_pretty_line()  # what a confirm box does mid-call
        yield _result("L1-4\none\n")

    await render_events(_events())
    out = capsys.readouterr().out
    assert out == ("\n* Read 'a.txt' \n\n* Read 'a.txt' L1-4 (done)\n\n")
    assert "\x1b[" not in out


def test_clear_streamed_lines_clears_exactly_the_given_rows(capsys: pytest.CaptureFixture[str]) -> None:
    """Two rows means two clears; the cursor lands on the topmost cleared row."""
    _clear_streamed_lines(2)
    assert capsys.readouterr().out == "\r\x1b[2K\x1b[1A\r\x1b[2K"


def test_clear_streamed_lines_ignores_non_positive_counts(capsys: pytest.CaptureFixture[str]) -> None:
    _clear_streamed_lines(0)
    _clear_streamed_lines(-1)
    assert capsys.readouterr().out == ""


def test_line_count_for_clear_counts_wrapped_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A streamed line wider than the terminal owns more than one row."""
    monkeypatch.setenv("COLUMNS", "20")
    assert _line_count_for_clear("assistant:", "x" * 20) == 2  # label + full row
    assert _line_count_for_clear("assistant:", "x" * 21) == 3  # label + two rows
    assert _line_count_for_clear("assistant:", "a\nb") == 3
    assert _line_count_for_clear("", "") == 0


async def test_flush_markdown_erases_wrapped_assistant_rows(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A wrapped assistant segment is fully erased before the markdown re-render."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    monkeypatch.setattr("plyngent.cli.display.markdown_render_available", lambda: True)
    monkeypatch.setenv("COLUMNS", "20")
    set_markdown_enabled(True)
    try:
        await render_events(
            _aiter(
                [
                    TextDeltaEvent(content="x" * 21),
                    _pretty_call("read_file", '{"path": "a.txt"}'),
                    _result("L1-4\none\n"),
                ]
            )
        )
    finally:
        set_markdown_enabled(True)
    out = capsys.readouterr().out
    # label + two wrapped body rows + the blank separator above the label.
    assert out.count("\x1b[2K") == 4
    assert out.count("\x1b[1A") == 3


async def test_pretty_prefix_closed_before_error_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Another event ends an open prefix line instead of appending to it."""
    monkeypatch.setattr("plyngent.cli.display.interactive_terminal", lambda: True)
    await render_events(
        _aiter(
            [
                _read_call('{"path": "a.txt"}'),
                ErrorEvent(message="boom", source="tool"),
                _result("error: tool 'read_file' failed: boom"),
            ]
        )
    )
    out = capsys.readouterr().out
    assert "* Read 'a.txt' \n" in out
    assert "[error] source=tool boom\n" in out
    assert "* Read 'a.txt' (error)\n" in out
