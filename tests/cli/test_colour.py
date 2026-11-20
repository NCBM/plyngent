"""Colour: which style each kind of line is written in, and the three ways to want none of it.

The claim this file is built from is narrow and it is the whole acceptance: a role decides how a line
looks and never what it says, so the same screen read as bytes is the same screen whether it was painted
or not. A painted stream is exercised through prompt_toolkit's own plain-text output, which is the
honest stand-in for a terminal that does not exist in a test (E90, E92).
"""

from __future__ import annotations

import io
from typing import get_args, override

import pytest

from plyngent.__main__ import _colour
from plyngent.cli.render import STYLES, Role, Shown
from plyngent.cli.screen import Sink


class _Terminal(io.StringIO):
    """A stream that says it is a terminal, and is otherwise a buffer — the flag's own question."""

    @override
    def isatty(self) -> bool:
        return True


def test_every_role_has_a_style_and_no_style_is_invented() -> None:
    roles = set(get_args(Role.__value__))

    assert set(STYLES) == roles, "the table is keyed by the whole closed set, so a ninth role is a decision"
    assert len(set(STYLES.values())) < len(roles), (
        "two kinds of bookkeeping may look alike; more colours would be noise"
    )


def test_painting_changes_the_style_and_never_the_text() -> None:
    plain, painted, refused = io.StringIO(), io.StringIO(), io.StringIO()

    Sink(plain).say(Shown("same words", "assistant"))
    Sink(painted, colour=True).say(Shown("same words", "assistant"))
    Sink(refused, colour=True).say(Shown("same words", "refusal"))

    # Prompt_toolkit's own writer turns a newline into the terminal's line ending; that is the terminal's
    # convention and not the text's, so the comparison is made on the text (E90).
    assert plain.getvalue() == _text(painted) == _text(refused) == "same words\n", (
        "a role is how a line looks; what it says is the event's"
    )


def _text(sink: io.StringIO) -> str:
    """What a painted stream said, with the terminal's line ending read back as a newline."""
    return sink.getvalue().replace("\r\n", "\n")


def test_a_run_that_asked_for_no_colour_gets_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert _colour(asked=True, stream=_Terminal()) is False


def test_a_stream_that_is_not_a_terminal_is_never_painted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert _colour(asked=False, stream=io.StringIO()) is False, "an escape code in a file is mojibake"


def test_the_environment_can_ask_for_no_colour(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")

    assert _colour(asked=False, stream=_Terminal()) is False


def test_an_empty_no_colour_is_a_shell_accident_rather_than_a_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """The convention is about a value, not about a name: an exported empty variable says nothing."""
    monkeypatch.setenv("NO_COLOR", "")

    assert _colour(asked=False, stream=_Terminal()) is True


def test_a_terminal_that_said_nothing_gets_its_colours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert _colour(asked=False, stream=_Terminal()) is True
