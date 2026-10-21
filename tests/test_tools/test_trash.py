from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import send2trash

from plyngent.tools import trash
from plyngent.tools.trash import is_available, move_to_trash, under_system_temp

# The freedesktop implementation is the only one whose trash location can be
# redirected for a test; macOS and Windows move into their own system trash.
freedesktop_only = pytest.mark.skipif(sys.platform != "linux", reason="freedesktop trash only")


@pytest.fixture
def home_trash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the freedesktop implementation at a trash under *tmp_path*.

    ``plat_other`` reads its trash locations from module globals computed at
    import time and picks "home" vs "external volume" by device number, so both
    are pinned here — otherwise a ``/tmp`` on a filesystem of its own would
    create a real trash directory on that volume.
    """
    plat_other = pytest.importorskip("send2trash.plat_other")
    root = tmp_path / "Trash"
    monkeypatch.setattr(plat_other, "XDG_DATA_HOME", os.fsencode(str(tmp_path)))
    monkeypatch.setattr(plat_other, "HOMETRASH_B", os.fsencode(str(root)))
    monkeypatch.setattr(plat_other, "HOMETRASH", str(root))
    monkeypatch.setattr(plat_other, "get_dev", lambda _path: 1)
    # A GIO-backed install would otherwise bypass every patch above.
    monkeypatch.setattr(send2trash, "send2trash", plat_other.send2trash)
    return root


def test_the_library_is_installed() -> None:
    assert is_available()


@freedesktop_only
def test_move_to_trash_lands_in_the_trash(home_trash: Path, tmp_path: Path) -> None:
    target = tmp_path / "gone.txt"
    _ = target.write_text("x")

    assert move_to_trash(target) is None

    assert not target.exists()
    assert (home_trash / "files" / "gone.txt").read_text() == "x"
    entry = (home_trash / "info" / "gone.txt.trashinfo").read_text()
    assert entry.startswith("[Trash Info]")
    assert "Path=" in entry
    assert "DeletionDate=" in entry


@freedesktop_only
def test_move_to_trash_moves_a_directory_in_one_piece(home_trash: Path, tmp_path: Path) -> None:
    target = tmp_path / "box"
    target.mkdir()
    _ = (target / "inner.txt").write_text("x")

    assert move_to_trash(target) is None

    assert not target.exists()
    assert (home_trash / "files" / "box" / "inner.txt").read_text() == "x"


@freedesktop_only
def test_trashing_twice_keeps_both_copies(home_trash: Path, tmp_path: Path) -> None:
    first = tmp_path / "dup.txt"
    second = tmp_path / "sub" / "dup.txt"
    second.parent.mkdir()
    _ = first.write_text("one")
    _ = second.write_text("two")

    assert move_to_trash(first) is None
    assert move_to_trash(second) is None

    names = sorted(path.name for path in (home_trash / "files").iterdir())
    assert names == ["dup 1.txt", "dup.txt"]


def test_a_refused_volume_names_the_escape_hatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "gone.txt"
    _ = target.write_text("x")

    def _refuse(_paths: object) -> None:
        raise send2trash.TrashPermissionError(str(target))

    monkeypatch.setattr(send2trash, "send2trash", _refuse)

    error = move_to_trash(target)

    assert error is not None
    assert error.startswith("error: cannot move to trash:")
    assert "permanent=true" in error
    assert target.is_file()


def test_an_oserror_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "gone.txt"
    _ = target.write_text("x")

    def _fail(_paths: object) -> None:
        raise OSError("disk on fire")

    monkeypatch.setattr(send2trash, "send2trash", _fail)

    assert move_to_trash(target) == "error: cannot move to trash: disk on fire"
    assert target.is_file()


def test_a_broken_install_leaves_the_file_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "gone.txt"
    _ = target.write_text("x")
    monkeypatch.setattr(trash, "is_available", lambda: False)

    error = move_to_trash(target)

    assert error is not None
    assert "send2trash is not installed" in error
    assert "permanent=true" in error
    assert target.is_file()


def test_under_system_temp(tmp_path: Path) -> None:
    assert under_system_temp(tmp_path)
    assert under_system_temp(tmp_path / "nested" / "file.txt")
    assert not under_system_temp(Path("/plyngent-not-temp/deep/file.txt"))
