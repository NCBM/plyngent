"""A count and its noun, written the way a person reads it — "1 tools" is a bug, not a number.

No story says this on its own: every tool that counts something says a sentence around the number, so
the only way to see the singular through a story is to arrange a group with exactly one tool in it. The
rule is one function, so it is tested as one.
"""

from __future__ import annotations

import pytest

from plyngent.kernel.words import counted, plural


@pytest.mark.parametrize(
    ("count", "noun", "said"),
    [
        (0, "tool", "tools"),
        (1, "tool", "tool"),
        (2, "tool", "tools"),
        (1, "entry", "entry"),
        (3, "entry", "entries"),
        (2, "day", "days"),
    ],
)
def test_the_noun_follows_the_count_and_nothing_else(count: int, noun: str, said: str) -> None:
    assert plural(count, noun) == said


def test_a_phrase_puts_the_number_in_front_of_the_noun_it_counted() -> None:
    assert counted(1, "path") == "1 path"
    assert counted(3, "path") == "3 paths"
