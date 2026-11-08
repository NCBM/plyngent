"""A delete the human can undo: the platform's trash, or a refusal that destroys nothing."""

from __future__ import annotations

import pytest
import send2trash

from plyngent.kernel import trash
from plyngent.kernel.host import Host


def test_the_trash_is_available_because_it_is_a_declared_dependency() -> None:
    assert trash.available(), "a missing send2trash is a broken install, not an expected absence"


def test_a_path_that_the_trash_takes_is_gone_from_where_it_was(tmp_path) -> None:
    doomed = tmp_path / "old.txt"
    _ = doomed.write_text("still wanted later")

    assert trash.to_trash(doomed) is None
    assert not doomed.exists()


def test_a_trash_that_refuses_leaves_the_file_alone(tmp_path, monkeypatch) -> None:
    kept = tmp_path / "kept.txt"
    _ = kept.write_text("x")

    def refuse(path: str) -> None:
        raise OSError(5, "no trash directory on this volume")

    monkeypatch.setattr(send2trash, "send2trash", refuse)

    trouble = trash.to_trash(kept)

    assert trouble is not None
    assert trouble.kind is trash.Kind.REFUSED, "a volume that will not take this path is about the path"
    assert "no trash directory" in trouble.said
    assert kept.exists(), "what the human was told was recoverable is not destroyed instead"


def test_a_declared_absence_of_the_trash_module_is_reported_not_destroyed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(trash, "available", lambda: False)

    trouble = trash.to_trash(tmp_path / "x")

    assert trouble is not None
    assert trouble.kind is trash.Kind.ABSENT, "a missing library is about the installation, not the path"
    assert trouble.said == "cannot move to trash: send2trash is not installed"


@pytest.mark.parametrize(("given", "expected"), [(True, True), (False, False)])
def test_the_trash_switch_belongs_to_the_host(tmp_path, *, given: bool, expected: bool) -> None:
    assert Host(tmp_path, trash_enabled=given).trash_enabled is expected
    assert Host(tmp_path).trash_enabled is True, "recoverable is the default, and the only safe one"
