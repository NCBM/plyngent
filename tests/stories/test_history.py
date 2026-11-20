"""Reading a conversation back: what happened here, in what order, and what continues what.

A projection answers what a model is given — one chain, and only the rows a model reads. What a person
asks is the other question, and it is the one thing a stored journal can answer that a screen cannot:
every row is still there, the bookkeeping included. The read that answers it carries the parent each row
names, so the shape of the conversation travels with the rows, at no question per branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stories.model import Held, text

from .driver import SESSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_the_entries_are_shown_in_the_order_they_were_written(story_with: Scripted) -> None:
    """The word `/history`, which is why the read exists: what is in this conversation, and what is not.

    The first row is the tool set this client attached, which the session wrote down about itself: the
    journal answers what happened, not what was said, and the two are not the same list.
    """
    story = await story_with(text("hello there"))
    await story.type("say something")
    await story.finished()
    read_from = len(story.notices())

    await story.type("/history")
    await story.settle()

    assert story.notices()[read_from:] == (
        "e1 state-change",
        "e2 user: say something",
        "e3 assistant: hello there",
        "e4 turn-ended",
    ), "one line per row, named by the id a client continues from — the turn's own end included"


async def test_a_turn_that_was_stopped_says_so_where_the_log_is_read(story_with: Scripted) -> None:
    """A turn's end is a row and not only an event, which is what a log read afterwards has to go on.

    Nothing else in the journal distinguishes a turn that finished from one whose rows simply stop: the
    rows of a process that was killed end at the same place a turn's do when it ends quietly. So the end
    is written down, below the turn's last row, and a person reading the log finds it there — with the
    reason, because "it ended" and "somebody stopped it" are two different endings.
    """
    story = await story_with(Held(text("too late")))
    await story.type("something slow")
    await story.press("ctrl-c")
    await story.until("turn 1 finished: user")
    read_from = len(story.notices())

    await story.type("/history")
    await story.settle()

    assert story.notices()[read_from:] == (
        "e1 state-change",
        "e2 user: something slow",
        "e3 turn-ended: user",
    ), "the row says where the turn ended, and why"


async def test_a_conversation_nothing_was_said_in_yet_reads_as_nothing(story_with: Scripted) -> None:
    """Empty is an answer: a word that printed nothing would be indistinguishable from one that failed."""
    story = await story_with(text("hello"))
    await story.type("/new --session work")
    read_from = len(story.notices())

    await story.type("/history --session work")
    await story.settle()

    assert story.notices()[read_from:] == ("this conversation has nothing in it yet",)


async def test_the_whole_forest_can_be_read_with_its_parents(story_with: Scripted) -> None:
    """What no line shows: each row says which one it continues, so the shape travels with the rows.

    A list of entries is a log; the same list with a parent on each row is the forest, and it is what a
    client needs to offer somebody the places they could go on from — without a second question per
    branch. The row nothing continues is where a turn grows, which is why the count is in the answer.
    """
    story = await story_with(text("first"), text("second"))
    await story.type("one")
    await story.finished()
    await story.type("two")
    await story.finished(turn=2)

    rows = await story.client.trajectory(SESSION)

    assert [row.id for row in rows] == ["e1", "e2", "e3", "e4", "e5", "e6", "e7"], (
        "arrival order, as the journal holds it"
    )
    assert [row.kind for row in rows] == [
        "state-change",
        "user",
        "assistant",
        "turn-ended",
        "user",
        "assistant",
        "turn-ended",
    ], "two turns, and each of them says where it ended"
    assert [row.begin_after for row in rows] == [None, None, "e2", "e3", "e4", "e5", "e6"], (
        "each row names the one it continues, so the chain is in the answer"
    )
    assert [row.preview for row in rows] == ["", "one", "first", "", "two", "second", ""], (
        "one line of what each row says, and nothing for a row that has no words of its own"
    )
    assert [row.children for row in rows] == [0, 1, 1, 1, 1, 1, 0], (
        "nothing continues the last one, which is where a turn grows"
    )


async def test_a_conversation_this_kernel_is_not_serving_has_no_rows(story_with: Scripted) -> None:
    """A read about a conversation nobody opened is answered with nothing, and nothing is invented."""
    story = await story_with(text("hello"))
    await story.type("say something")
    await story.finished()

    assert await story.client.trajectory("elsewhere") == ()


async def test_travelling_moves_where_the_next_message_continues(story_with: Scripted) -> None:
    """Standing on an entry an earlier answer grew from: the next message goes past it, not past the end.

    Nothing is asked of the kernel to do this — which branch a frontend reads is its own business — so
    what the story can see is the consequence: the conversation the model is given is the one that ends
    at that entry, plus what was just said.
    """
    story = await story_with(text("first"), text("second"))
    await story.type("one")
    await story.finished()
    read_from = len(story.notices())

    await story.type("/travel --entry e2")
    await story.settle()
    assert story.notices()[read_from:] == ("continuing from e2",), "the cursor moved, and says where to"

    await story.type("two")
    await story.finished(turn=2)

    assert [message.text for message in story.requests()[-1].messages] == ["one", "two"], (
        "the answer that grew from e2 is not on the chain the new message continues"
    )


async def test_travelling_to_an_entry_that_is_not_there_says_so_and_stays(story_with: Scripted) -> None:
    """A name typed by a person is a guess: a place this conversation does not have is a line, not a cursor."""
    story = await story_with(text("first"), text("second"))
    await story.type("one")
    await story.finished()
    read_from = len(story.notices())

    await story.type("/travel --entry e99")
    await story.settle()
    assert story.notices()[read_from:] == ("no entry e99 in this conversation; /history lists the ones there are",)

    await story.type("two")
    await story.finished(turn=2)
    assert [message.text for message in story.requests()[-1].messages] == ["one", "first", "two"], (
        "and the conversation still continues from where it did"
    )
