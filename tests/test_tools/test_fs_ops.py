from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.tools.file import copy_path, delete_path, fs_ops, move_path, read_file, write_file
from tests.test_tools.helpers import call_sync


def test_copy_file(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "a.txt", "hello")
    out = call_sync(copy_path, "a.txt", "b.txt")
    assert "copied" in out
    assert call_sync(read_file, "b.txt") == "L1-1\nhello"
    assert (workspace / "a.txt").is_file()


def test_copy_directory(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "src/x.txt", "x")
    _ = call_sync(write_file, "src/nested/y.txt", "y")
    out = call_sync(copy_path, "src", "dst")
    assert "directory" in out
    assert call_sync(read_file, "dst/x.txt") == "L1-1\nx"
    assert call_sync(read_file, "dst/nested/y.txt") == "L1-1\ny"


def test_copy_overwrite(workspace: object) -> None:
    del workspace
    _ = call_sync(write_file, "a.txt", "old")
    _ = call_sync(write_file, "b.txt", "new")
    assert "exists" in call_sync(copy_path, "b.txt", "a.txt")
    out = call_sync(copy_path, "b.txt", "a.txt", overwrite=True)
    assert "copied" in out
    assert call_sync(read_file, "a.txt") == "L1-1\nnew"


def test_move_file(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "old.txt", "data")
    out = call_sync(move_path, "old.txt", "renamed.txt")
    assert "moved" in out
    assert not (workspace / "old.txt").exists()
    assert call_sync(read_file, "renamed.txt") == "L1-1\ndata"


def test_move_directory(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "d/a.txt", "a")
    out = call_sync(move_path, "d", "e")
    assert "moved" in out
    assert not (workspace / "d").exists()
    assert call_sync(read_file, "e/a.txt") == "L1-1\na"


def test_delete_file(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "t.txt", "x")
    out = call_sync(delete_path, "t.txt")
    assert "deleted" in out
    assert not (workspace / "t.txt").exists()


def test_delete_empty_dir(workspace: object) -> None:
    assert isinstance(workspace, Path)
    (workspace / "empty").mkdir()
    out = call_sync(delete_path, "empty")
    assert "deleted" in out
    assert not (workspace / "empty").exists()


def test_delete_nonempty_requires_recursive(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "box/a.txt", "a")
    out = call_sync(delete_path, "box")
    assert "not empty" in out
    assert (workspace / "box" / "a.txt").is_file()
    # The workspace fixture lives under the system temp dir, so it is removed
    # for good rather than trashed (see the trash tests below).
    out2 = call_sync(delete_path, "box", recursive=True)
    assert "recursive" in out2
    assert not (workspace / "box").exists()


def test_delete_workspace_root_forbidden(workspace: object) -> None:
    del workspace
    out = call_sync(delete_path, ".")
    assert "workspace root" in out


def test_move_workspace_root_forbidden(workspace: object) -> None:
    del workspace
    out = call_sync(move_path, ".", "elsewhere")
    assert "workspace root" in out


def test_missing_source(workspace: object) -> None:
    del workspace
    assert "does not exist" in call_sync(copy_path, "nope", "x")
    assert "does not exist" in call_sync(move_path, "nope", "x")
    assert "does not exist" in call_sync(delete_path, "nope")


def _elsewhere_than_temp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the workspace is not scratch space, so deletes use the trash."""
    monkeypatch.setattr(fs_ops, "under_system_temp", lambda _path: False)


def test_a_temp_path_is_deleted_for_good(workspace: object) -> None:
    assert isinstance(workspace, Path)
    _ = call_sync(write_file, "t.txt", "x")
    out = call_sync(delete_path, "t.txt")
    assert "trash" not in out
    assert not (workspace / "t.txt").exists()


def test_delete_goes_to_the_trash(workspace: object, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(workspace, Path)
    _elsewhere_than_temp(monkeypatch)
    trashed: list[Path] = []
    monkeypatch.setattr(fs_ops, "move_to_trash", trashed.append)
    _ = call_sync(write_file, "t.txt", "x")

    out = call_sync(delete_path, "t.txt")

    assert out == "deleted file t.txt (moved to system trash)"
    assert [path.name for path in trashed] == ["t.txt"]


def test_recursive_delete_goes_to_the_trash(workspace: object, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(workspace, Path)
    _elsewhere_than_temp(monkeypatch)
    trashed: list[Path] = []
    monkeypatch.setattr(fs_ops, "move_to_trash", trashed.append)
    _ = call_sync(write_file, "box/a.txt", "a")

    out = call_sync(delete_path, "box", recursive=True)

    assert out == "deleted directory box (moved to system trash)"
    assert [path.name for path in trashed] == ["box"]


def test_a_refused_trash_keeps_the_path(workspace: object, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(workspace, Path)
    _elsewhere_than_temp(monkeypatch)
    refused = "error: cannot move to trash: nope (use permanent=true)"
    monkeypatch.setattr(fs_ops, "move_to_trash", lambda _target: refused)
    _ = call_sync(write_file, "t.txt", "x")

    out = call_sync(delete_path, "t.txt")

    assert out == refused
    assert (workspace / "t.txt").is_file()


def test_permanent_delete_skips_the_trash(workspace: object, monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(workspace, Path)

    def _never(_target: Path) -> str | None:
        msg = "a permanent delete must not touch the trash"
        raise AssertionError(msg)

    monkeypatch.setattr(fs_ops, "move_to_trash", _never)
    _ = call_sync(write_file, "t.txt", "x")

    out = call_sync(delete_path, "t.txt", permanent=True)

    assert out == "deleted file t.txt"
    assert not (workspace / "t.txt").exists()
