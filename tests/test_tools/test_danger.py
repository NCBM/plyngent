from __future__ import annotations

from pathlib import Path

from plyngent.tools.danger import classify_danger


def test_classify_delete_and_move() -> None:
    assert classify_danger("delete_path", {"path": "a.txt"}) == "delete path 'a.txt'"
    assert "recursively" in (classify_danger("delete_path", {"path": "d", "recursive": True}) or "")
    assert "move" in (classify_danger("move_path", {"src": "a", "dst": "b"}) or "")


def test_classify_copy_overwrite_only() -> None:
    assert classify_danger("copy_path", {"src": "a", "dst": "b"}) is None
    assert "overwrite" in (classify_danger("copy_path", {"src": "a", "dst": "b", "overwrite": True}) or "")


def test_classify_write_file_overwrite(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = (workspace / "x.txt").write_text("old", encoding="utf-8")
    assert "overwrite" in (classify_danger("write_file", {"path": "x.txt", "content": "n"}) or "")
    assert classify_danger("write_file", {"path": "new.txt", "content": "n"}) is None


def test_classify_safe_tools() -> None:
    assert classify_danger("read_file", {"path": "a"}) is None
    assert classify_danger("run_command", {"command": ["echo", "hi"]}) is None
