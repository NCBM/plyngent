from __future__ import annotations

from pathlib import Path

from plyngent.tools.file import tree
from tests.test_tools.helpers import call_sync


def _build_sample(root: Path) -> None:
    _ = (root / "a.txt").write_text("a", encoding="utf-8")
    _ = (root / ".hidden_file").write_text("h", encoding="utf-8")
    (root / "src").mkdir()
    _ = (root / "src" / "main.py").write_text("m", encoding="utf-8")
    (root / "src" / "nested").mkdir()
    _ = (root / "src" / "nested" / "deep.txt").write_text("d", encoding="utf-8")
    (root / ".hidden_dir").mkdir()
    _ = (root / ".hidden_dir" / "secret.txt").write_text("s", encoding="utf-8")
    (root / ".git").mkdir()
    _ = (root / ".git" / "config").write_text("g", encoding="utf-8")
    (root / "pkg").mkdir()
    for i in range(60):
        _ = (root / "pkg" / f"f{i:02d}.txt").write_text("x", encoding="utf-8")


def test_tree_basic_skips_vcs_and_hidden_dirs(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, ".")
    assert "a.txt" in out
    assert ".hidden_file" in out  # hidden *files* still shown
    assert "src/" in out
    assert "main.py" in out
    assert ".git" not in out
    assert ".hidden_dir" not in out
    assert "secret.txt" not in out


def test_tree_include_hidden_dirs(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, ".", skip_hidden_dirs=False)
    assert ".hidden_dir/" in out
    assert "secret.txt" in out
    assert ".git" not in out  # VCS still always skipped


def test_tree_max_depth(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, ".", max_depth=1)
    assert "src/" in out
    assert "main.py" not in out
    assert "nested" not in out


def test_tree_max_entries(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, "pkg", max_depth=2, max_entries=10)
    assert "more entries not shown" in out
    # only first 10 of 60 files listed as entries
    listed = [line for line in out.splitlines() if line.strip().endswith(".txt")]
    assert len(listed) == 10


def test_tree_origin_subdir(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, "src")
    assert out.startswith("src/")
    assert "main.py" in out
    assert "deep.txt" in out


def test_tree_not_directory(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _build_sample(workspace)
    out = call_sync(tree, "a.txt")
    assert "error" in out


def test_tree_invalid_limits(workspace: object) -> None:
    del workspace
    assert "max_depth" in call_sync(tree, ".", max_depth=0)
    assert "max_entries" in call_sync(tree, ".", max_entries=0)


def test_tree_default_noise_dirs(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / "src").mkdir()
    _ = (workspace / "src" / "app.py").write_text("x", encoding="utf-8")
    (workspace / "node_modules").mkdir()
    _ = (workspace / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    _ = (workspace / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")
    out = call_sync(tree, ".")
    assert "src/" in out
    assert "app.py" in out
    assert "node_modules" not in out
    assert "__pycache__" not in out


def test_tree_skip_dirs_empty_shows_noise(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / "node_modules").mkdir()
    _ = (workspace / "node_modules" / "pkg.js").write_text("x", encoding="utf-8")
    out = call_sync(tree, ".", skip_dirs=[])
    assert "node_modules/" in out
    assert "pkg.js" in out


def test_tree_skip_dirs_custom(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / "keep_me").mkdir()
    _ = (workspace / "keep_me" / "a.txt").write_text("x", encoding="utf-8")
    (workspace / "drop_me").mkdir()
    _ = (workspace / "drop_me" / "b.txt").write_text("x", encoding="utf-8")
    # Custom list replaces default noise set (node_modules not in list → would show if present).
    out = call_sync(tree, ".", skip_dirs=["drop_me"])
    assert "keep_me/" in out
    assert "drop_me" not in out


def test_tree_path_denylist_walk(workspace: object) -> None:
    from plyngent.tools.workspace import set_path_denylist

    assert isinstance(workspace, Path)
    (workspace / "ok").mkdir()
    _ = (workspace / "ok" / "a.txt").write_text("x", encoding="utf-8")
    (workspace / "secrets").mkdir()
    _ = (workspace / "secrets" / "key.txt").write_text("x", encoding="utf-8")
    set_path_denylist(["/secrets"])
    try:
        out = call_sync(tree, ".")
        assert "ok/" in out
        assert "secrets" not in out
        out2 = call_sync(tree, ".", apply_path_denylist=False, skip_dirs=[])
        assert "secrets/" in out2
    finally:
        set_path_denylist(None)
