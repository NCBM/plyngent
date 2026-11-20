"""Where a conversation is kept: beside the workspace, and never inside it.

A workspace is a tree a person works in. Opening a conversation over it may not leave anything in it —
no database, no sidecar, no directory — because the first thing a person does with such a tree is commit
it. The history therefore belongs to a data directory of its own, one file per workspace, and a story
names that file so a test's history lands beside the test's workspace, not in the person's own
data.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.journal import MEMORY_JOURNAL, journal_path
from plyngent.kernel.model import Assistant, Reply, User
from plyngent.kernel.model import Usage as Cost
from plyngent.kernel.trail import Usage as Spent
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    type Scripted = Callable[..., Awaitable[Story]]


async def test_a_conversation_is_kept_beside_the_workspace_and_leaves_it_alone(tmp_path: Path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    story = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([text("noted")]), journal=journal
    )
    try:
        before = set(story.workspace.iterdir())
        await story.type("a first thought")

        assert story.screen() == ("turn 1 started (chat)", "noted", "turn 1 finished")
        assert set(story.workspace.iterdir()) == before, "the conversation was kept out of the workspace"
        assert journal.exists(), "and it was kept where the story said, not in the person's data directory"
    finally:
        await story.close()


async def test_a_restarted_client_continues_the_conversation_it_left(tmp_path: Path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root, transport=transport, model=ScriptedModel([text("the first answer")]), journal=journal
    )
    try:
        await first.type("a first thought")

        assert first.screen() == ("turn 1 started (chat)", "the first answer", "turn 1 finished")
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root, transport=transport, model=ScriptedModel([text("the second answer")]), journal=journal
    )
    try:
        await second.type("and another")

        assert second.screen() == ("turn 2 started (chat)", "the second answer", "turn 2 finished"), (
            "the turn ids resume too: the log allocates them, so a restart continues the count"
        )
        (asked,) = second.requests()
        assert asked.messages == (
            User(text="a first thought"),
            Assistant(text="the first answer"),
            User(text="and another"),
        ), "the model was given the conversation the process that ended had written"
    finally:
        await second.close()


async def test_what_a_conversation_spent_outlives_the_process_that_spent_it(tmp_path: Path, transport: str) -> None:
    """The tally a turn says aloud is read once and gone; a price is added up afterwards, offline."""
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([Reply(text="noted", usage=Cost(prompt=11, completion=2, cached=5))]),
        journal=journal,
    )
    try:
        await first.type("a first thought")
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(root, transport=transport, model=ScriptedModel([]), journal=journal)
    try:
        rows = [
            row.payload for row in await second.host.trail.entries(second.session) if isinstance(row.payload, Spent)
        ]

        assert [(row.turn, row.prompt, row.completion, row.cached) for row in rows] == [(1, 11, 2, 5)], (
            "a second process reads what the first one spent, three counts and nothing to infer"
        )
    finally:
        await second.close()


async def test_a_listing_says_where_each_conversation_ends(story_with: Scripted) -> None:
    story = await story_with(text("one"))

    await story.type("a thought")
    await story.type("/new --session work")
    await story.type("/sessions")

    listed = story.screen()[-1]
    assert re.fullmatch(r"sessions: main \(.+?\), work", listed), (
        f"the conversation with history says where it ends, and the empty one says nothing: {listed!r}"
    )


async def test_a_conversation_that_was_closed_is_still_listed(story_with: Scripted) -> None:
    """Closing is "this kernel stops serving it", not "it is gone": the journal still holds it.

    The set a listing answers is the file's, so a conversation somebody closed an hour ago is still the
    one they are looking for — and it still says where it ends, because that is what the file has.
    """
    story = await story_with(text("one"))
    await story.type("a thought")
    await story.type("/new --session work --seed-from main --seed-from e3")
    await story.type("/close --session work")

    await story.type("/sessions")

    listed = story.screen()[-1]
    assert re.fullmatch(r"sessions: main \(.+?\), work \(.+?\)", listed), (
        f"both are in the file, and both say where they end: {listed!r}"
    )


async def test_a_conversation_nobody_meant_to_keep_leaves_nothing(tmp_path: Path, transport: str) -> None:
    story = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([text("noted")]), journal=MEMORY_JOURNAL
    )
    try:
        await story.type("a passing thought")

        assert story.screen()[-1] == "turn 1 finished"
        assert set(story.workspace.iterdir()) == set(), "and a journal that lasts one process makes no file"
        assert not journal_path(story.workspace).exists(), "not even where this workspace's history would go"
    finally:
        await story.close()
