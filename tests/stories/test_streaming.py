"""An answer that arrives in pieces, and what a person reads while it is still coming.

The kernel says a fragment as it arrives and then the whole message the fragments add up to; a frontend
has both and only one screen. So this is the one place where "what was said" and "what a person read"
are not the same list of events, and what a story can watch is the screen: which lines are on it, and
when each of them got there (§0.13, E94).

The model is held between two pieces in the second story on purpose. "A line appears as soon as it is
whole" is invisible once the turn has ended — the finished screen looks the same either way — so the only
way to assert it is to look at the screen while the model is still talking.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent_creative.model import Streamed, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_an_answer_in_pieces_is_read_as_the_lines_it_makes(story_with: Scripted) -> None:
    said = "one two\nthree\nfour"
    story = await story_with(Streamed(pieces=("one ", "two\nthr", "ee\nfour"), reply=text(said)))

    await story.type("say something")
    await story.finished()

    assert story.screen() == (
        "turn 1 started (chat)",
        "one two",
        "three",
        "four",
        "turn 1 finished",
    ), "the breaks are the model's, not the network's — and the answer was not printed twice"


async def test_a_whole_line_is_read_while_the_answer_is_still_coming(story_with: Scripted) -> None:
    ready = asyncio.Event()
    story = await story_with(
        Streamed(pieces=("first line\nhalf a ", "second"), reply=text("first line\nhalf a second"), ready=ready),
    )

    await story.type("say something")
    await story.until("first line")

    assert "half a" not in story.screen(), "a fragment is held until the newline that makes a line of it"
    ready.set()
    await story.finished()

    assert story.screen()[1:] == ("first line", "half a second", "turn 1 finished"), (
        "the tail nobody finished is closed by the turn ending, and it is a line of its own"
    )


async def test_a_terminal_reads_an_answer_as_the_markdown_it_is(story_with: Scripted) -> None:
    """The same two pieces, on a screen: a heading is a heading and a list is a list (E100)."""
    said = "# Title\n\n- one\n- two\n"
    story = await story_with(Streamed(pieces=("# Title\n\n- one\n", "- two\n"), reply=text(said)), markdown=True)

    await story.type("say something")
    await story.finished()

    read = "\n".join(story.screen())
    assert "Title" in read, "markdown is a format, so what reaches the screen is not the marker"
    assert "#" not in read
    assert " • one" in read and " • two" in read, "the list is laid out as a list"
