"""``@path`` mentions: parsing, token spelling, and Tab completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.mentions import (
    complete_mention_paths,
    mention_completions,
    mention_token_for,
    parse_mentions,
)

if TYPE_CHECKING:
    from pathlib import Path


def _paths(line: str) -> list[str]:
    return [mention.path for mention in parse_mentions(line)]


def test_a_mention_is_a_whitespace_delimited_token() -> None:
    assert _paths("look at @shot.png please") == ["shot.png"]
    assert _paths("@a.png and @b/c.txt") == ["a.png", "b/c.txt"]
    assert _paths("@a.png\n@b.png") == ["a.png", "b.png"]


def test_an_email_address_is_not_a_mention() -> None:
    assert _paths("mail me at someone@example.com") == []


def test_a_quoted_mention_may_hold_spaces() -> None:
    assert _paths('look at @"screens/my shot.png"') == ["screens/my shot.png"]
    assert _paths('@"a b.png" and @c.png') == ["a b.png", "c.png"]


def test_an_unterminated_quote_is_left_alone() -> None:
    assert _paths('still typing @"my sh') == []


def test_a_bare_at_sign_is_not_a_mention() -> None:
    assert _paths("@ and @@@") == []


def test_a_token_quotes_a_path_with_spaces() -> None:
    assert mention_token_for("shot.png") == "@shot.png"
    assert mention_token_for("my shot.png") == '@"my shot.png"'


def test_completion_lists_a_directory(tmp_path: Path) -> None:
    (tmp_path / "shot.png").write_bytes(b"")
    (tmp_path / "screens").mkdir()

    completions = complete_mention_paths("sc", cwd=tmp_path)

    assert completions == ["screens/"]


def test_completion_continues_through_a_directory(tmp_path: Path) -> None:
    (tmp_path / "screens").mkdir()
    (tmp_path / "screens" / "shot.png").write_bytes(b"")

    assert complete_mention_paths("screens/sh", cwd=tmp_path) == ["screens/shot.png"]
    assert complete_mention_paths("screens/", cwd=tmp_path) == ["screens/shot.png"]


def test_completion_hides_dot_files_unless_asked(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("x", encoding="utf-8")

    assert complete_mention_paths("", cwd=tmp_path) == []
    assert complete_mention_paths(".", cwd=tmp_path) == [".env"]


def test_completion_ignores_a_missing_directory(tmp_path: Path) -> None:
    assert complete_mention_paths("nope/x", cwd=tmp_path) == []


def test_completions_keep_the_at_sign_and_quote_spaces(tmp_path: Path) -> None:
    (tmp_path / "my shot.png").write_bytes(b"")
    (tmp_path / "shots").mkdir()

    assert mention_completions("@my", cwd=tmp_path) == ['@"my shot.png"']
    assert mention_completions("@shots", cwd=tmp_path) == ["@shots/"]
    assert mention_completions("@", cwd=tmp_path) == ['@"my shot.png"', "@shots/"]


def test_mention_completions_ignores_a_plain_word() -> None:
    assert mention_completions("shot") == []
