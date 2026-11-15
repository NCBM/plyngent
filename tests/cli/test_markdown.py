"""Markdown, as the two pure steps between what the model wrote and the rows a screen shows.

Nothing here knows about a terminal: :func:`chunks` cuts the text where it can no longer change, and
:func:`lines` lays one block out for a width it is handed. So what a test can pin is exactly what E100
decided — a blank line is the boundary, a fence is not, an unclosed marker stays literal, and a width is
an argument rather than something measured.
"""

from __future__ import annotations

from plyngent.cli.markdown import chunks, lines


def test_a_blank_line_is_where_a_block_stops() -> None:
    closed, tail = chunks("one\n\ntwo\n")

    assert closed == ("one",), "everything above the blank line is settled"
    assert tail == "two\n", "and what is below it is the block still being written"


def test_nothing_closes_while_the_block_is_still_being_written() -> None:
    assert chunks("one\ntwo") == ((), "one\ntwo"), "no blank line yet, so there is no block to settle"


def test_a_blank_line_inside_a_fence_is_part_of_the_code() -> None:
    closed, tail = chunks("```python\nx = 1\n\ny = 2\n```\n\nafter\n")

    assert closed == ("```python\nx = 1\n\ny = 2\n```",), "a blank line in a code block is a blank line in code"
    assert tail == "after\n"


def test_an_unclosed_fence_keeps_everything_after_it() -> None:
    """The block a window draws may be a fence that is not finished, and nothing in it may be closed early."""
    closed, tail = chunks("```\ncode\n\nmore\n")

    assert closed == ()
    assert tail == "```\ncode\n\nmore\n"


def test_blank_lines_with_nothing_between_them_are_not_blocks() -> None:
    assert chunks("\n\n\ntext") == ((), "text"), "a screen does not print the blank lines of a blank line"


def test_a_list_is_laid_out_as_bullets_without_the_markers() -> None:
    assert lines("- one\n- two", width=40) == (" • one", " • two"), "the layout is rich's, and the marker is gone"


def test_a_paragraph_is_wrapped_to_the_width_it_was_given() -> None:
    rows = lines(" ".join(["word"] * 12), width=20)

    assert all(len(row) <= 20 for row in rows), "the width arrives as an argument, so nothing measures a screen"
    assert len(rows) > 1
    assert " ".join(rows).split() == ["word"] * 12, "and every word is still there, in order"


def test_a_screen_that_reports_nothing_still_gets_a_layout() -> None:
    assert lines(" ".join(["word"] * 40), width=0), "a terminal that answers zero columns is not a screen with no room"


def test_a_row_is_laid_out_in_cells_and_not_in_characters() -> None:
    """A wide character takes two columns, so a row of them that fits holds half as many: rich counts
    cells, which is the one measurement this layer is not allowed to make itself (E92)."""
    rows = lines("中文" * 40, width=20)

    assert max(len(row) for row in rows) <= 10, "twenty cells is ten of these characters, and not twenty"


def test_markdown_that_was_never_closed_is_read_as_the_text_it_is() -> None:
    assert lines("**open bold", width=40) == ("**open bold",), "half a marker is not a marker, and nothing is guessed"
