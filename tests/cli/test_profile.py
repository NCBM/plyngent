"""The frontend's own memory: one pair in one file, read and written without ceremony.

A profile is not a configuration surface — nothing here can name an endpoint or a cap — so these tests are
about the three things a small file can get wrong: a missing one, a torn one, and one that says a route
and a model that nothing else agrees with. The first two are the same answer to a caller, and the third is
the caller's to check (E172).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.profile import NAME, Profile, read, write

if TYPE_CHECKING:
    from pathlib import Path


def test_a_profile_nobody_wrote_is_nothing_remembered(tmp_path: Path) -> None:
    assert read(tmp_path / NAME) == Profile(), "no file, no memory: the same answer as a torn one"


def test_a_profile_that_cannot_be_read_is_nothing_remembered(tmp_path: Path) -> None:
    path = tmp_path / NAME
    _ = path.write_text("this is not toml = = =", encoding="utf-8")

    assert read(path) == Profile(), "a convenience file may not hold a run hostage"


def test_a_pair_survives_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "deeper" / NAME

    write(path, Profile(route="local", model="big"))

    assert read(path) == Profile(route="local", model="big")
    assert 'model = "big"' in path.read_text(encoding="utf-8"), "what it says is what a person can read"


def test_a_file_that_says_something_else_says_nothing(tmp_path: Path) -> None:
    """A table this build does not know is not a guess: the pair is the whole of what the file means."""
    path = tmp_path / NAME
    _ = path.write_text('[last]\nroute = 7\nmodel = "big"\n', encoding="utf-8")

    assert read(path) == Profile(model="big"), "the word that is not a name is dropped, and the other kept"
