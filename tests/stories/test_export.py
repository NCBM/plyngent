"""The conversation written down: what a model would be given, as text a person can keep.

An export is a projection, so there is nothing to see move — what a story can check is that the text is
the conversation and that the handle it answers with is the way back to it. The handle is read back from
the store, because that is where a handle points; going through a tool instead would test the tool.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent_creative.model import text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: The line an export earns on the screen: the handle it was kept under, and how much text it was.
KEPT = re.compile(r"document (d\d+) kept \((\d+) chars\)")

#: What a listing says about a session that has a head: the name, and where it ends.
LISTED = re.compile(r"sessions: main \((e\d+)\)")


async def test_an_export_keeps_the_conversation_as_the_text_a_model_would_read(story_with: Scripted) -> None:
    story = await story_with(text("the first answer"), text("the second answer"))
    await story.type("a first thought")
    await story.type("a second thought")

    await story.type("/export")

    held = await _exported(story)
    assert held == (
        "## person\n\na first thought\n\n"
        "## assistant\n\nthe first answer\n\n"
        "## person\n\na second thought\n\n"
        "## assistant\n\nthe second answer"
    ), "every message on the chain, once, with who said it"


async def test_an_export_of_nothing_writes_nothing(story_with: Scripted) -> None:
    story = await story_with(text("unused"))
    await story.type("/export")

    assert story.screen() == ("export refused: there is nothing to export: this conversation has no messages yet",), (
        "an empty conversation has nothing to write down, and saying so is not an error"
    )
    assert not story.requests(), "and nothing was asked of the model"


async def test_an_export_of_a_branch_stops_where_it_was_told(story_with: Scripted) -> None:
    story = await story_with(text("one"), text("two"))
    await story.type("a first thought")
    await story.type("/sessions")
    found = LISTED.fullmatch(story.screen()[-1])
    assert found is not None, f"the listing did not name a head: {story.screen()[-1:]!r}"
    await story.type("a second thought")

    await story.type(f"/export --begin-after {found[1]}")

    held = await _exported(story)
    assert "a first thought" in held
    assert "a second thought" not in held, "a branch is written as far as it goes, and no further"


async def _exported(story: Story) -> str:
    """What the export on this screen kept, read out of the store it went into."""
    kept = KEPT.fullmatch(story.screen()[-1])
    assert kept is not None, f"the export says where it went: {story.screen()[-1:]!r}"
    found = await story.host.documents.get(story.session, kept[1])
    assert found is not None, f"{kept[1]} was said to be kept and is not there"
    return found.data.decode()
