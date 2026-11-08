"""An ignore rule is text with a small grammar: what it says about one path is all this module knows."""

from __future__ import annotations

from pathlib import Path

import pytest

from plyngent.kernel.ignore import DEFAULT_IGNORE_FILES, IgnoreSources, Patterns, Rules


def _decide(lines: list[str], path: str, *, is_dir: bool = False) -> bool | None:
    return Patterns.parse(lines).decide(path, is_dir=is_dir)


def test_a_name_without_a_separator_matches_at_any_depth() -> None:
    assert _decide(["build"], "build") is True
    assert _decide(["build"], "a/b/build") is True, "an unanchored rule is about the name, not the place"
    assert _decide(["*.log"], "logs/server.log") is True
    assert _decide(["*.log"], "logs/server.txt") is None, "no rule spoke about it"


def test_a_separator_anchors_the_rule_to_the_directory_that_carried_it() -> None:
    assert _decide(["/build"], "build") is True
    assert _decide(["/build"], "a/build") is None, "the leading slash means the root and nothing deeper"
    assert _decide(["src/*.py"], "src/a.py") is True
    assert _decide(["src/*.py"], "a/src/a.py") is None


def test_a_wildcard_never_crosses_a_separator_but_a_double_one_does() -> None:
    assert _decide(["src/*.py"], "src/deep/a.py") is None
    assert _decide(["src/**/*.py"], "src/deep/a.py") is True
    assert _decide(["src/**/*.py"], "src/a.py") is True, "** covers no directory at all as well"
    assert _decide(["a/**"], "a/b/c") is True
    assert _decide(["?at"], "cat") is True
    assert _decide(["[bc]at"], "bat") is True
    assert _decide(["[!bc]at"], "bat") is None, "a negated class is the inverse of a class"


def test_a_trailing_slash_is_about_directories_and_is_handed_to_the_walk() -> None:
    assert _decide(["build/"], "build", is_dir=True) is True
    assert _decide(["build/"], "build", is_dir=False) is None, "a file called build is not a directory"


def test_the_last_rule_that_matches_wins_and_bang_takes_it_back() -> None:
    rules = ["*.log", "!keep.log"]

    assert _decide(rules, "server.log") is True
    assert _decide(rules, "keep.log") is False
    assert _decide(["!keep.log", "*.log"], "keep.log") is True, "order is the whole of it"


def test_comments_and_blank_lines_say_nothing_but_a_comment_can_be_escaped() -> None:
    assert _decide(["", "   ", "# build", "build"], "build") is True
    assert _decide(["\\#build"], "#build") is True
    assert _decide(["build "], "build") is True, "trailing spaces are not part of the name"


def test_a_source_that_cannot_be_read_says_the_same_thing_as_one_that_is_not_there(tmp_path) -> None:
    assert Patterns.read(tmp_path / "absent") is None
    _ = (tmp_path / ".gitignore").write_text("build/\n!build/keep\n")
    found = Patterns.read(tmp_path / ".gitignore")
    assert found is not None
    assert found.rewinds, "the walk may look inside an excluded directory because of the bang"


def test_sources_are_listed_and_a_later_name_overrules_an_earlier_one(tmp_path) -> None:
    sources = IgnoreSources(files=(".gitignore", ".rgignore"))
    _ = (tmp_path / ".gitignore").write_text("*.log\n")
    _ = (tmp_path / ".rgignore").write_text("!keep.log\n")

    found = sources.read(tmp_path)

    assert found.decide("keep.log", is_dir=False) is False, "the later source had the last word"


def test_a_call_can_name_the_sources_it_reads_and_none_is_a_real_answer(tmp_path) -> None:
    sources = IgnoreSources(files=(".gitignore",))
    _ = (tmp_path / ".gitignore").write_text("*.log\n")
    _ = (tmp_path / ".ignore").write_text("*.tmp\n")

    assert sources.read(tmp_path, [".ignore"]).decide("x.tmp", is_dir=False) is True
    assert sources.read(tmp_path, []).decide("x.tmp", is_dir=False) is None, "this call read no source at all"
    assert sources.read(tmp_path).decide("x.log", is_dir=False) is True, "and the declaration is the default"


def test_the_workspace_patterns_belong_to_no_file() -> None:
    sources = IgnoreSources(patterns=("*.pem",))

    assert sources.policy.decide("secrets/server.pem", is_dir=False) is True
    assert sources.read(Path("/nonexistent")) == Patterns(), "policy is not read from a directory"


def test_a_workspace_keeps_back_names_and_sources_as_one_value() -> None:
    rules = Rules()

    assert rules.hidden == frozenset({"CVS", "_darcs"}), "the names that are metadata without a dot"
    assert rules.ignore.files == DEFAULT_IGNORE_FILES, "and the sources every checkout is read with"
    assert rules.ignore.patterns == (), "a checkout's own answer is not the host's"


def test_a_pattern_that_is_not_a_regex_is_dropped_rather_than_crashing_a_listing() -> None:
    assert _decide(["a["], "a[") is True
    assert _decide(["a["], "a") is None


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("docs/**/notes.md", "docs/notes.md", True),
        ("docs/**/notes.md", "docs/a/b/notes.md", True),
        ("**/node_modules", "a/node_modules", True),
        ("**/node_modules", "node_modules", True),
        ("*.py", ".hidden.py", True),
    ],
)
def test_the_grammar_holds_for_the_shapes_rules_are_actually_written_in(
    pattern: str, path: str, *, expected: bool
) -> None:
    assert _decide([pattern], path) is expected
