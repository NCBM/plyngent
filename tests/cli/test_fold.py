"""What a stream of events looks like as the lines and the window a person reads.

:func:`~plyngent.cli.render.render` is a pure function of one event; the fold is not, because "is this a
line yet" is a question about what has already reached the screen. So it is tested for what it is: a
small machine that is handed events in order, and asked what went out between them.

The two lanes are the two halves of it. A lane that cannot move its cursor reads **lines** — a fragment
waits for a newline, and a tail is committed by whatever goes on the screen next. A terminal reads
**markdown** — a block is cut at a blank line, rendered once, and the block still being written is the
window the screen redraws in place (E100). What is asserted here is only what a story cannot see: the
moments between two lines, which of the two messages the kernel sends for one answer is the one a person
reads, and where a window becomes a line.
"""

from __future__ import annotations

from plyngent.cli.render import WINDOW_ROWS, Fold, Shown
from plyngent.wire.event import AssistantDelta, AssistantText, Batch, TurnFinished


def _delta(text: str, *, turn: int = 1, session: str = "main") -> AssistantDelta:
    """One fragment of a turn's answer, as the kernel says it."""
    return AssistantDelta(session=session, turn=turn, text=text)


def test_a_line_goes_out_when_it_is_whole_and_not_before() -> None:
    fold = Fold()

    assert fold.push(_delta("half a ")).lines == (), "a fragment is not a line: there is no newline in it"
    assert fold.push(_delta("")).lines == (), "an empty fragment says nothing at all"
    assert fold.push(_delta("sentence")).lines == ()
    assert fold.push(_delta("\nnext")).lines == (Shown("half a sentence", "assistant"),), (
        "the line is the fragments added up, and it goes out the moment the newline arrives"
    )


def test_a_lane_with_no_cursor_never_has_a_window() -> None:
    """Nothing on a stream can be taken back, so a fragment that is not a line is not on the screen either."""
    fold = Fold()

    assert fold.push(_delta("half a sentence")).window == ()


def test_the_message_the_fragments_add_up_to_is_not_said_twice() -> None:
    fold = Fold()

    _ = fold.push(_delta("said once\n"))

    assert fold.push(AssistantText(session="main", turn=1, text="said once\n")).lines == (), (
        "the fragments already were the lines, so the whole message is dropped, not printed again"
    )


def test_a_tail_nobody_finished_is_committed_by_the_next_line_on_the_screen() -> None:
    fold = Fold()

    _ = fold.push(_delta("half a sentence"))

    assert fold.push(TurnFinished(turn=1, reason="")).lines == (
        Shown("half a sentence", "assistant"),
        Shown("turn 1 finished", "notice"),
    ), "the unclosed tail is a line of its own, and nothing is left unsaid behind the turn"


def test_a_second_turn_is_read_after_the_first_one_streamed() -> None:
    fold = Fold()

    _ = fold.push(_delta("one"))
    _ = fold.push(AssistantText(session="main", turn=1, text="one"))

    assert fold.push(AssistantText(session="main", turn=2, text="two")).lines == (Shown("two", "assistant"),), (
        "only the message that is the whole of what was already said is dropped, and that is per turn"
    )


def test_a_batch_is_folded_the_way_its_items_would_have_been() -> None:
    fold = Fold()

    assert fold.push(Batch(items=(_delta("one\n"), _delta("two")))).lines == (Shown("one", "assistant"),), (
        "coalescing is a transport fact: a fragment in a batch is held exactly like one on its own"
    )


def test_a_line_from_another_session_is_named_like_every_line_of_that_session() -> None:
    fold = Fold()

    assert fold.push(_delta("here\n", session="other"), focus="main").lines == (Shown("[other] here", "assistant"),)


def test_a_block_that_closed_is_final_and_the_one_being_written_is_the_window() -> None:
    fold = Fold(markdown=True)

    started = fold.push(_delta("a paragraph\n"))
    assert started.lines == (), "nothing is final while the block it is in is still open"
    assert started.window == (Shown("a paragraph", "assistant"),), "but a person reads it all the same"

    closed = fold.push(_delta("\nand a second"))
    assert closed.lines == (Shown("a paragraph", "assistant"),), "the blank line is what settles a block"
    assert closed.window == (Shown("and a second", "assistant"),), "the next block is the new window"


def test_the_message_the_fragments_add_up_to_is_not_said_twice_on_a_screen_either() -> None:
    fold = Fold(markdown=True)

    _ = fold.push(_delta("said once"))
    drawn = fold.push(AssistantText(session="main", turn=1, text="said once"))

    assert drawn.lines == (Shown("said once", "assistant"),), "the block became final when the turn said it"
    assert drawn.window == (), "and the answer is on the screen once, not once per event"


def test_two_blocks_are_set_apart_the_way_the_blank_line_between_them_was() -> None:
    fold = Fold(markdown=True)

    drawn = fold.push(_delta("first\n\nsecond\n\n"), width=60)

    assert drawn.lines == (
        Shown("first", "assistant"),
        Shown("", "assistant"),
        Shown("second", "assistant"),
    ), "each block is rendered on its own, so the blank line between them is written here"


def test_a_window_too_tall_to_redraw_is_committed_and_the_block_starts_again() -> None:
    fold = Fold(markdown=True)

    drawn = fold.push(_delta(" ".join(["word"] * 200)), width=40)

    assert drawn.window == (), "a block past the window's height is not redrawn in place any more"
    assert len(drawn.lines) > WINDOW_ROWS, "it is committed as it stands, which is the price of growing one"
    assert fold.push(_delta("more")).window == (Shown("more", "assistant"),), "and the next fragment opens a new one"


def test_the_width_a_block_is_laid_out_for_is_the_width_it_is_given() -> None:
    narrow = Fold(markdown=True).push(_delta("one two three four five"), width=12).window
    wide = Fold(markdown=True).push(_delta("one two three four five"), width=80).window

    assert len(narrow) > len(wide), "the screen's own width decides the wrapping: this layer never measures"
