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
    from plyngent.tools.workspace import clear_workspace_root, set_workspace_root

    set_workspace_root(tmp_path)
    try:
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
    finally:
        clear_workspace_root()


def test_classify_safe_tools() -> None:
    assert classify_danger("read_file", {"path": "a"}) is None
    assert classify_danger("listdir", {"path": "."}) is None
    assert classify_danger("run_command", {"command": ["echo", "hi"]}) is None
    assert classify_danger("open_pty", {"command": ["true"]}) is None
    assert classify_danger("run_command", {"command": ["ls", "-la"]}) is None


def test_classify_run_command_batch_risky() -> None:
    reason = classify_danger(
        "run_command_batch",
        {
            "commands": [
                {"command": ["echo", "ok"]},
                {"command": ["bash", "-c", "echo risky"]},
            ]
        },
    )
    assert reason is not None
    assert "run_command_batch" in reason
    assert "risky" in reason or "bash" in reason
    assert (
        classify_danger(
            "run_command_batch",
            {"commands": [{"command": ["echo", "ok"]}]},
        )
        is None
    )


def test_classify_shell_and_dash_c() -> None:
    r = classify_danger("run_command", {"command": ["bash", "-c", "rm -rf /"]})
    assert r is not None
    assert "bash -c" in r
    assert "rm -rf" in r

    r2 = classify_danger("run_command", {"command": ["python3", "-c", "print(1)"]})
    assert r2 is not None
    assert "python" in r2
    assert "-c" in r2
    assert "print(1)" in r2

    r_py = classify_danger("run_command", {"command": ["python", "-c", "import os"]})
    assert r_py is not None and "python -c" in r_py

    r3 = classify_danger("open_pty", {"command": ["bash"]})
    assert r3 is not None and "bash" in r3
    assert "interactive" in r3 or "review" in r3

    r4 = classify_danger("open_pty", {"command": ["python3"]})
    assert r4 is not None and "python" in r4

    r5 = classify_danger("open_pty", {"command": ["python3", "-c", "x=1"]})
    assert r5 is not None and "-c" in r5


async def test_confirm_deny_with_comment() -> None:
    from plyngent.agent.tools import ToolRegistry, tool
    from plyngent.tools.danger import classify_danger as danger

    @tool
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
    script = "line1" + "
" + "line2" + "
" + "line3"
    reason = classify_danger(
        "run_command",
        {"command": ["bash", "-c", script]},
    )
    assert reason is not None
    assert "$(command)" in reason
    assert "line1" in reason and "line2" in reason and "line3" in reason
    argv_line = next(ln for ln in reason.splitlines() if "argv:" in ln)
    assert "line1" not in argv_line
    lines = reason.splitlines()
    idx = lines.index("  command:")
    body = lines[idx + 1 :]
    assert body
    assert all(ln.startswith("  ") for ln in body)
