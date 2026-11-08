"""``<key>`` markers become the bytes a terminal would send — and a typo becomes an error."""

from __future__ import annotations

import pytest

from plyngent.kernel.pty.keys import KeySyntaxError, decode_write_data


@pytest.mark.parametrize(
    ("written", "sent"),
    [
        ("<enter>", "\r"),
        ("<return>", "\r"),
        ("<esc>", "\x1b"),
        ("<tab>", "\t"),
        ("<s-tab>", "\x1b[Z"),
        ("<backspace>", "\x7f"),
        ("<f5>", "\x1b[15~"),
        ("<c-a>", "\x01"),
        ("<c-[>", "\x1b"),
        ("<m-x>", "\x1bx"),
        ("<c-up>", "\x1b[1;5A"),
        ("<s-home>", "\x1b[1;2H"),
        ("<c-s-f5>", "\x1b[15;6~"),
        ("<up*3>", "\x1b[A\x1b[A\x1b[A"),
        ("<c-space>", "\x00"),
    ],
)
def test_a_marker_becomes_the_sequence_the_terminal_would_send(written: str, sent: str) -> None:
    assert decode_write_data(written) == sent


def test_plain_text_is_typed_as_it_is() -> None:
    assert decode_write_data("press enter") == "press enter"
    assert decode_write_data("sort < file") == "sort < file", "a `<` that opens nothing stays a `<`"
    assert decode_write_data("<<") == "<", "and this is how a literal one is written"
    assert decode_write_data("a < b > c") == "a < b > c", "a marker carries no whitespace"


@pytest.mark.parametrize(
    ("written", "sent"),
    [
        (r"\n", "\n"),
        (r"\r", "\r"),
        (r"\t", "\t"),
        (r"\e", "\x1b"),
        (r"\E", "\x1b"),
        (r"\0", "\x00"),
        (r"\x1b", "\x1b"),
        (r"\u00e9", "é"),
        (r"\q", r"\q"),
    ],
)
def test_an_escape_is_a_character_the_caller_cannot_type(written: str, sent: str) -> None:
    assert decode_write_data(written) == sent


@pytest.mark.parametrize("written", ["<c-uop>", "<nope>", "<up*0>", "<s-a>", "<c-1>"])
def test_a_marker_that_names_nothing_is_refused_rather_than_typed(written: str) -> None:
    with pytest.raises(KeySyntaxError):
        _ = decode_write_data(written)


def test_an_empty_write_sends_nothing() -> None:
    assert decode_write_data("") == ""
