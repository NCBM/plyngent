from __future__ import annotations

from pathlib import Path

from plyngent.tools.file.glob_paths import glob_paths
from tests.test_tools.helpers import call_sync


def test_glob_paths_basic(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = (workspace / "a.py").write_text("x", encoding="utf-8")
    (workspace / "sub").mkdir()
    _ = (workspace / "sub" / "b.py").write_text("y", encoding="utf-8")
    _ = (workspace / "c.txt").write_text("z", encoding="utf-8")
    out = call_sync(glob_paths, "**/*.py")
    assert "a.py" in out
    assert "sub/b.py" in out
    assert "c.txt" not in out


def test_glob_paths_skip_git(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / ".git").mkdir()
    _ = (workspace / ".git" / "config").write_text("x", encoding="utf-8")
    _ = (workspace / "ok.py").write_text("x", encoding="utf-8")
    out = call_sync(glob_paths, "**/*")
    assert "ok.py" in out
    assert ".git" not in out


def test_glob_paths_max_matches(workspace: object) -> None:
    assert isinstance(workspace, Path)
    for i in range(5):
        _ = (workspace / f"f{i}.txt").write_text("x", encoding="utf-8")
    out = call_sync(glob_paths, "*.txt", max_matches=2)
    assert "truncated" in out
    assert out.count("\n") >= 1


def test_glob_paths_empty_pattern(workspace: object) -> None:
    del workspace
    assert "error" in call_sync(glob_paths, "")
