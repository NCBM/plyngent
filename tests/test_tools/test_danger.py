from __future__ import annotations

from pathlib import Path

from plyngent.tools.danger import classify_danger


def test_classify_delete_and_move() -> None:
    assert classify_danger("delete_path", {"path": "a.txt"}) == "delete path 'a.txt'"
    assert "recursively" in (classify_danger("delete_path", {"path": "d", "recursive": True}) or "")
    assert "move" in (classify_danger("move_path", {"src": "a", "dst": "b"}) or "")


def test_classify_copy() -> None:
    # Without overwrite flag, copy is not soft-confirmed.
    assert classify_danger("copy_path", {"src": "a", "dst": "b"}) is None
    assert classify_danger("copy_path", {"src": "a", "dst": "b", "overwrite": False}) is None


def test_classify_write_overwrite_only(tmp_path: Path) -> None:
    from plyngent.tools.context import InstanceState, bind_instance
    from plyngent.tools.workspace import set_workspace_root

    instance = InstanceState(workspace_root=tmp_path.resolve())
    with bind_instance(instance):
        _ = set_workspace_root(tmp_path)
        # New file: no soft-confirm
        assert classify_danger("write_file", {"path": "new.txt"}) is None
        # Partial edits: never soft-confirm
        assert classify_danger("edit_replace", {"path": "x.txt"}) is None
        assert classify_danger("edit_lineno", {"path": "x.txt", "start_line": 1, "end_line": 2}) is None
        # Existing file write: confirm total overwrite
        (tmp_path / "x.txt").write_text("old", encoding="utf-8")
        reason = classify_danger("write_file", {"path": "x.txt"})
        assert reason is not None and "overwrite" in reason
        (tmp_path / "dst.txt").write_text("d", encoding="utf-8")
        (tmp_path / "src.txt").write_text("s", encoding="utf-8")
        assert classify_danger("copy_path", {"src": "src.txt", "dst": "dst.txt", "overwrite": False}) is None
        creason = classify_danger("copy_path", {"src": "src.txt", "dst": "dst.txt", "overwrite": True})
        assert creason is not None and "overwrite" in creason


def test_classify_safe_tools() -> None:
    assert classify_danger("read_file", {"path": "a"}) is None
    assert classify_danger("listdir", {"path": "."}) is None
    assert classify_danger("run_argv", {"argv": ["echo", "hi"]}) is None
    assert classify_danger("open_pty", {"command": ["true"]}) is None
    assert classify_danger("run_argv", {"argv": ["ls", "-la"]}) is None


def test_classify_run_argv_batch_risky() -> None:
    reason = classify_danger(
        "run_argv_batch",
        {
            "steps": [
                {"argv": ["echo", "ok"]},
                {"argv": ["bash", "-c", "echo risky"]},
            ]
        },
    )
    assert reason is not None
    assert "run_argv_batch" in reason
    assert "risky" in reason or "bash" in reason
    assert (
        classify_danger(
            "run_argv_batch",
            {"steps": [{"argv": ["echo", "ok"]}]},
        )
        is None
    )


def test_classify_shell_and_dash_c() -> None:
    r = classify_danger("run_argv", {"argv": ["bash", "-c", "rm -rf /"]})
    assert r is not None
    assert "bash -c" in r
    assert "rm -rf" in r

    r2 = classify_danger("run_argv", {"argv": ["python3", "-c", "print(1)"]})
    assert r2 is not None
    assert "python" in r2
    assert "-c" in r2
    assert "print(1)" in r2

    r_py = classify_danger("run_argv", {"argv": ["python", "-c", "import os"]})
    assert r_py is not None and "python -c" in r_py

    r3 = classify_danger("open_pty", {"command": ["bash"]})
    assert r3 is not None and "bash" in r3
    assert "interactive" in r3 or "review" in r3

    r4 = classify_danger("open_pty", {"command": ["python3"]})
    assert r4 is not None and "python" in r4

    r5 = classify_danger("open_pty", {"command": ["python3", "-c", "x=1"]})
    assert r5 is not None and "-c" in r5


def test_dash_c_means_code_only_for_interpreters() -> None:
    # ``-c`` is a plain flag elsewhere: no code to review, no confirm.
    assert classify_danger("run_argv", {"argv": ["grep", "-c", "needle", "f.txt"]}) is None
    assert classify_danger("run_argv", {"argv": ["od", "-c", "f.bin"]}) is None
    assert classify_danger("run_argv", {"argv": ["sort", "-c", "f.txt"]}) is None


def test_interpreter_review_does_not_split_on_dash_c() -> None:
    script = classify_danger("run_argv", {"argv": ["python", "script.py"]})
    one_liner = classify_danger("run_argv", {"argv": ["python", "-c", "print(1)"]})
    bare_shell = classify_danger("open_pty", {"command": ["bash"]})
    assert script is not None and "interpreter 'python'" in script
    assert one_liner is not None and "interpreter 'python'" in one_liner
    assert bare_shell is not None and "interpreter 'bash'" in bare_shell
    # Only the ``-c`` form has code to print below the argv line.
    assert "command:" not in script
    assert "command:" in one_liner
    assert "print(1)" in one_liner


def test_versioned_interpreter_is_still_reviewed() -> None:
    reason = classify_danger("run_argv", {"argv": ["python3.12", "-c", "import os"]})
    assert reason is not None and "interpreter 'python3.12'" in reason
    assert "import os" in reason


def test_wrappers_reach_the_interpreter() -> None:
    for argv in (
        ["env", "FOO=1", "python", "-c", "print(1)"],
        ["sudo", "-u", "root", "python", "-c", "print(1)"],
        ["timeout", "5", "bash", "-c", "rm -rf /"],
        ["pdm", "run", "python", "-c", "print(1)"],
        ["uv", "run", "--with", "rich", "python", "-c", "print(1)"],
    ):
        reason = classify_danger("run_argv", {"argv": argv})
        assert reason is not None, argv
        assert "interpreter 'python'" in reason or "interpreter 'bash'" in reason
        assert "$(command)" in reason


def test_value_free_flags_do_not_hide_the_interpreter() -> None:
    """``exec -c`` takes no value, so the interpreter behind it is still reviewed."""
    reason = classify_danger("run_argv", {"argv": ["exec", "-c", "python", "-c", "print(1)"]})
    assert reason is not None
    assert "interpreter 'python'" in reason
    assert "print(1)" in reason


def test_only_the_interpreter_dash_c_becomes_a_placeholder() -> None:
    reason = classify_danger("run_argv", {"argv": ["ionice", "-c", "2", "python", "-c", "print(1)"]})
    assert reason is not None
    argv_line = next(line for line in reason.splitlines() if line.startswith("  argv:"))
    assert "ionice -c 2" in argv_line
    assert argv_line.count("$(command)") == 1
    assert "print(1)" in reason


def test_detached_runner_is_always_reviewed() -> None:
    assert classify_danger("run_argv", {"argv": ["sleep", "5"]}) is None
    reason = classify_danger("run_argv", {"argv": ["nohup", "sleep", "5"]})
    assert reason is not None and "nohup (detached run)" in reason
    nested = classify_danger("run_argv", {"argv": ["nohup", "python", "-c", "print(1)"]})
    assert nested is not None
    assert "nohup (detached run)" in nested and "interpreter 'python'" in nested
    assert "print(1)" in nested


async def test_confirm_deny_with_comment() -> None:
    from plyngent.agent.tools import ToolRegistry, tool
    from plyngent.tools.danger import classify_danger as danger

    @tool(register=False)
    def delete_path(path: str) -> str:
        return f"deleted {path}"

    async def deny_comment(name: str, args: object, reason: str) -> str:
        del name, args, reason
        return "too destructive for this session"

    reg = ToolRegistry([delete_path], danger=danger, on_confirm=deny_comment)
    out = await reg.execute("delete_path", '{"path": "x"}')
    assert "denied" in out
    assert "user comment:" in out
    assert "too destructive" in out


def test_shell_confirm_formats_command_placeholder() -> None:
    script = chr(10).join(["line1", "line2", "line3"])
    reason = classify_danger(
        "run_argv",
        {"argv": ["bash", "-c", script]},
    )
    assert reason is not None
    assert "$(command)" in reason
    assert "line1" in reason and "line2" in reason and "line3" in reason
    argv_line = next(ln for ln in reason.splitlines() if ln.startswith("  argv:"))
    assert "line1" not in argv_line
    lines = reason.splitlines()
    idx = lines.index("  command:")
    body = lines[idx + 1 :]
    assert body
    assert all(ln.startswith("  ") for ln in body)
