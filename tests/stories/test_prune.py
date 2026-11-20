"""One prune, seen from the outside: a line on the screen and what the next turn is given.

A prune is placement, not deletion, so what a story can check is exactly what a person would: the kernel
says it wrote one, the next request no longer carries the words it named, and the rest of the conversation
is still there. That the row itself stayed is the journal's own story (`tests/test_kernel/test_prune.py`) —
the screen cannot show a thing that is deliberately not shown.

The entry to hide is read off a listing: ids are the journal's to allocate, and a
story that guessed one would be testing its own arithmetic.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.model import Note, User
from plyngent.kernel.trail import NOTICE_MARKER
from tests.stories.model import text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: The line a prune earns on the screen, with a count of what it hid.
WRITTEN = re.compile(r"prune e\d+ written \((\d+) withheld\)")

#: The line the elision itself earns, at the boundary where it is said to the model.
NOTICED = re.compile(r"notice e\d+ written: the person hid 1 earlier entry")

#: What a listing says about a session that has a head: the name, and where the conversation ends.
LISTED = re.compile(r"sessions: main \((e\d+)\)")

#: What the model is told: the marker, and the sentence — model-visible wording, so it is pinned here.
ELISION = (
    f"{NOTICE_MARKER}\n"
    "Hidden from you: 1 earlier entry of this conversation, put away by the person working with you, who "
    "said: said too much. What you read below is not the whole conversation, so do not assume that a "
    "decision or a result you cannot see did not happen."
)


async def test_a_prune_hides_what_it_names_from_the_next_turn(story_with: Scripted) -> None:
    story = await story_with(text("the first answer"), text("the second answer"))
    await story.type("a first thought")
    (entry,) = await _ending(story)

    await story.type(f'/prune --targets {entry} --reason "said too much"')
    await story.type("a second thought")

    assert any(WRITTEN.fullmatch(line) for line in story.screen()), f"the prune says what it hid: {story.screen()!r}"
    assert any(NOTICED.fullmatch(line) for line in story.screen()), (
        f"and the elision is said to the model at the next boundary: {story.screen()!r}"
    )
    told = story.requests()[-1].messages
    assert told == (User(text="a first thought"), User(text="a second thought"), Note(text=ELISION)), (
        "the answer it named is gone from what the model reads, the person's own words are not, and the hole "
        "is named — at the round's own boundary, so the line is the last thing the model reads before it answers"
    )


async def test_a_prune_that_names_nothing_writes_nothing(story_with: Scripted) -> None:
    story = await story_with(text("one"))
    await story.type("a first thought")
    asked = len(story.requests())

    await story.type("/prune")
    await story.type("/prune --targets e999")

    assert story.screen()[-2:] == (
        "prune refused: a prune with no entries hides nothing",
        "prune refused: no entry 'e999' in session main: a prune can only hide an entry that exists",
    ), "a prune that half happened would be a history nobody can account for, so a bad list writes nothing"
    assert len(story.requests()) == asked, "and nothing was written, so nothing was asked differently"


async def _ending(story: Story) -> tuple[str, ...]:
    """Where this session's conversation ends, read off the listing the way a person would."""
    await story.type("/sessions")
    found = LISTED.fullmatch(story.screen()[-1])
    assert found is not None, f"the listing did not name a head: {story.screen()[-1:]!r}"
    return (found[1],)
