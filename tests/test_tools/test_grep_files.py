from __future__ import annotations

from pathlib import Path

from plyngent.tools.file.grep_files import grep_files
from tests.test_tools.helpers import call_sync


def test_grep_files_basic(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = (workspace / "a.py").write_text("hello world\nfoo\n", encoding="utf-8")
    _ = (workspace / "b.py").write_text("nope\n", encoding="utf-8")
    out = call_sync(grep_files, "hello")
    assert "a.py:1:" in out
    assert "hello world" in out
    assert "b.py" not in out


def test_grep_files_case_insensitive(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = (workspace / "t.txt").write_text("Hello\n", encoding="utf-8")
    assert "no matches" in call_sync(grep_files, "hello")
    out = call_sync(grep_files, "hello", case_insensitive=True)
    assert "t.txt:1:" in out


def test_grep_files_invalid_regex(workspace: object) -> None:
    del workspace
    assert "invalid regex" in call_sync(grep_files, "[")


def test_grep_files_skip_git(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / ".git").mkdir()
    _ = (workspace / ".git" / "x").write_text("secret\n", encoding="utf-8")
    _ = (workspace / "ok.txt").write_text("secret\n", encoding="utf-8")
    out = call_sync(grep_files, "secret")
    assert "ok.txt" in out
    assert ".git" not in out


def test_grep_files_max_matches(workspace: object) -> None:
    assert isinstance(workspace, Path)
    lines = "\n".join(f"match {i}" for i in range(10))
    _ = (workspace / "m.txt").write_text(lines + "\n", encoding="utf-8")
    out = call_sync(grep_files, "match", max_matches=3)
    assert "truncated" in out
