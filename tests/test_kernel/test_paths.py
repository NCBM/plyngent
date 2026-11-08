"""A path has one identity, one spelling for people, and one key for the order of a batch."""

from __future__ import annotations

import os
from pathlib import Path

from plyngent.kernel import paths


def test_every_spelling_of_one_file_is_one_path(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    _ = (tmp_path / "b.txt").write_text("x")

    assert paths.normalize("b.txt", base=tmp_path) == tmp_path / "b.txt"
    assert paths.normalize("a/../b.txt", base=tmp_path) == tmp_path / "b.txt"
    assert paths.normalize(str(tmp_path / "b.txt"), base=tmp_path) == tmp_path / "b.txt"


def test_a_symlink_and_its_target_are_the_same_path(tmp_path) -> None:
    _ = (tmp_path / "real.txt").write_text("x")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")

    assert paths.normalize("link.txt", base=tmp_path) == paths.normalize("real.txt", base=tmp_path)


def test_a_home_directory_is_expanded_wherever_it_is_written(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    (home / "proj").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    assert paths.normalize("~/proj", base=tmp_path) == (home / "proj").resolve()


def test_two_spellings_of_one_file_are_one_lock_key(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    _ = (tmp_path / "b.txt").write_text("x")

    one = paths.key(paths.normalize("a/../b.txt", base=tmp_path))
    other = paths.key(paths.normalize(str(tmp_path / "b.txt"), base=tmp_path))

    assert one == other, "the batch graph and the lock table key on this, so it cannot depend on spelling"


def test_case_folding_is_the_platforms_answer(tmp_path) -> None:
    made = paths.normalize("A.txt", base=tmp_path)

    assert paths.key(made) == os.path.normcase(str(made)), "one rule, from the platform, in one place"


def test_a_path_under_the_workspace_is_written_relative_to_it(tmp_path) -> None:
    target = tmp_path / "src" / "a.py"

    assert paths.display(target, root=tmp_path) == "src/a.py"


def test_a_path_under_the_home_directory_says_home_when_it_is_shorter(tmp_path, monkeypatch) -> None:
    home = tmp_path / "home"
    workspace = home / "some" / "checked" / "out"
    target = home / "notes.txt"
    monkeypatch.setattr(paths, "_home", lambda: home)

    assert paths.display(target, root=workspace) == "~/notes.txt"


def test_a_path_that_is_under_neither_is_absolute(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_home", lambda: tmp_path / "home")

    assert (
        paths.display(tmp_path / "elsewhere" / "x.txt", root=tmp_path / "ws")
        == (tmp_path / "elsewhere" / "x.txt").as_posix()
    )


def test_display_is_a_pure_function(tmp_path) -> None:
    target = tmp_path / "a.txt"

    assert paths.display(target, root=tmp_path) == paths.display(target, root=tmp_path)
    assert paths.display(Path("/")) == "/"


def test_a_host_with_no_home_directory_still_has_absolute_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "_home", lambda: None)

    assert paths.display(tmp_path / "a.txt") == (tmp_path / "a.txt").as_posix()
