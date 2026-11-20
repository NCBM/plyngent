"""Starting a conversation from another one's text: what a compaction into a fresh session is.

The copy is the projection and nothing else — the messages a model would have been given — so it carries
no grants, no todo, no route and no name. That is the whole point of having it: a conversation can be
continued under a name of its own without inheriting anything a person agreed to in the other one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.wire.command import Submit
from plyngent.wire.event import TurnFinished
from tests.stories.model import text

from .driver import SESSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def _one_exchange(story_with: Scripted) -> Story:
    """A conversation with one thing said and one answer: three rows, of which two are messages."""
    story = await story_with(text("first"), text("second"))
    await story.type("one")
    await story.finished()
    return story


async def test_a_new_conversation_can_start_from_another_ones_text(story_with: Scripted) -> None:
    """The seeded session is a conversation of its own whose text is somebody else's, in order."""
    story = await _one_exchange(story_with)
    read_from = len(story.screen())

    await story.type("/new --session work --seed-from main --seed-from e3")
    await story.settle()
    assert story.screen()[read_from:] == ("[work] seeded 2 messages from main",)

    read_from = len(story.notices())
    await story.type("/history --session work")
    await story.settle()
    assert story.notices()[read_from:] == ("e5 user: one", "e6 assistant: first"), (
        "the same text, written down as rows of the new conversation"
    )


async def test_what_was_copied_is_what_the_next_turn_is_given(story_with: Scripted) -> None:
    """The copy is a history, not a note about one: a turn continues from it as from its own."""
    story = await _one_exchange(story_with)
    await story.type("/new --session work --seed-from main --seed-from e3")
    other = await story.another_client()
    rows = await other.trajectory("work")

    assert other.submit(Submit(session="work", text="two", begin_after=rows[-1].id))
    async for event in other.events():
        if isinstance(event, TurnFinished):
            break

    assert [message.text for message in story.requests()[-1].messages] == ["one", "first", "two"], (
        "the copied text came first, because it is that conversation's history"
    )


async def test_a_seed_naming_an_entry_that_is_not_there_copies_nothing(story_with: Scripted) -> None:
    """One act, all or nothing: a conversation half-filled with somebody else's text is unaccountable."""
    story = await _one_exchange(story_with)
    read_from = len(story.screen())

    await story.type("/new --session work --seed-from main --seed-from e9")
    await story.settle()

    assert story.screen()[read_from:] == (
        "[work] seed refused: no entry 'e9' in session main: only a row that exists can be copied from",
    )


async def test_a_seed_that_was_refused_leaves_no_conversation_behind(story_with: Scripted) -> None:
    """A refusal is not a half-open session: an empty conversation would be a worse answer than a reason."""
    story = await _one_exchange(story_with)
    read_from = len(story.screen())

    await story.type("/new --session work --seed-from main --seed-from e9")
    await story.type("/sessions")
    await story.settle()

    assert story.screen()[read_from:] == (
        "[work] seed refused: no entry 'e9' in session main: only a row that exists can be copied from",
        "sessions: main (e3)",
    ), "the refused conversation is not being served, and the one that was copied from still ends where it did"


async def test_a_conversation_that_is_already_open_is_not_seeded_again(story_with: Scripted) -> None:
    """Copying into a conversation a turn could be using would land beside that turn, not in it."""
    story = await _one_exchange(story_with)
    await story.type("/new --session work --seed-from main --seed-from e3")
    read_from = len(story.screen())

    await story.type("/new --session work --seed-from main --seed-from e3")
    await story.settle()

    assert story.screen()[read_from:] == (
        "[work] seed refused: work is already open here, so nothing was copied into it",
    )


async def test_a_seeded_conversation_does_not_inherit_what_was_agreed_in_the_other(story_with: Scripted) -> None:
    """The text travels; the settings do not. ``/grants`` is the one of them a person can see."""
    story = await _one_exchange(story_with)
    await story.type("/new --session work --seed-from main --seed-from e3")
    read_from = len(story.notices())

    await story.type("/grants --session work")
    await story.settle()

    assert story.notices()[read_from:] == ("nothing outside this workspace is allowed",)


async def test_the_conversation_the_text_came_from_is_untouched(story_with: Scripted) -> None:
    """A copy and not a move: the session that was copied is where it was, and says the same thing."""
    story = await _one_exchange(story_with)

    await story.type("/new --session work --seed-from main --seed-from e3")

    assert [row.preview for row in await story.client.trajectory(SESSION)] == ["", "one", "first", ""], (
        "main holds what it held, and the copy lives under its own name"
    )
