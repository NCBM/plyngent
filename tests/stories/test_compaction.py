"""Compaction: the one entry a summary is, and what a session does with it.

A summary is an ordinary entry with one rule attached — **where it continues is its whole meaning** — so
the stories here are about placement rather than about summarising: what a compaction writes (one row, no
turn), what a session says when it cannot write one, and where a restarted client continues once the
summary is the end of the conversation it can reach.

Nothing here asks a model to compress anything. The text arrives in the command, because the command is
the writer of this tag and generating a summary needs a provider (R11).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.model import Assistant, User
from plyngent.kernel.trail import Summary, as_message
from tests.stories.driver import Story
from tests.stories.model import Held, ScriptedModel, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    type Scripted = Callable[..., Awaitable[Story]]

#: The line a written summary earns on the screen: the entry it went to, and how much text stands in
#: for what it replaced. The id is the journal's to allocate, so it is matched rather than guessed.
WRITTEN = re.compile(r"summary e\d+ written \((\d+) chars\)")


async def test_compacting_writes_one_entry_and_no_turn(story_with: Scripted) -> None:
    story = await story_with(text("the first answer"))

    await story.type("a first thought")
    await story.type("/compact --text settled")

    assert story.screen()[:3] == ("turn 1 started (chat)", "the first answer", "turn 1 finished")
    assert len(story.screen()) == 4, "a compaction writes one line and starts no turn"
    written = WRITTEN.fullmatch(story.screen()[3])
    assert written is not None and written[1] == "7", f"the summary says how much it stands for: {story.screen()[3]!r}"
    assert len(story.requests()) == 1, "and it asks the model nothing: the text came with the command"


async def test_a_refused_compaction_says_why_and_writes_nothing(story_with: Scripted) -> None:
    story = await story_with(text("one"))
    await story.type("a thought")
    before = len(story.screen())

    await story.type("/compact")
    await story.type("/compact --begin-after e999 --text later")
    await story.type("/compact --session nowhere --text later")

    assert story.screen()[before:] == (
        "compact refused: a summary with no text compresses nothing",
        "compact refused: no entry 'e999' in session main: an entry can only continue one that exists",
        "[nowhere] compact refused: no such session here: open one first",
    ), "each refusal says what was wrong, and the journal's own sentence is the one passed on"
    assert len(story.requests()) == 1, "and nothing was written, so there was nothing new to ask about"


async def test_a_compaction_does_not_wait_for_a_running_turn(story_with: Scripted) -> None:
    held = Held(text("about time"))
    story = await story_with(held)

    await story.type("a first thought")
    await story.type("/compact --text settled")

    assert WRITTEN.fullmatch(story.screen()[-1]) is not None, (
        f"a summary is one row, so it does not need the session's turn slot: {story.screen()!r}"
    )

    held.ready.set()
    await story.settle()

    assert story.screen()[-2:] == ("about time", "turn 1 finished"), "and the turn it did not wait for ran on"


async def test_a_restarted_client_continues_at_the_summary(tmp_path: Path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root, transport=transport, model=ScriptedModel([text("the first answer")]), journal=journal
    )
    try:
        await first.type("a first thought")
        await first.type("/compact --text settled")

        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root, transport=transport, model=ScriptedModel([text("the second answer")]), journal=journal
    )
    try:
        await second.type("and another")

        (asked,) = second.requests()
        assert asked.messages == (as_message(Summary(text="settled")), User(text="and another")), (
            "the listing said the conversation ends at the summary, so that is where the next message went"
        )
    finally:
        await second.close()


async def test_the_model_is_given_the_summary_and_not_what_it_replaced(story_with: Scripted) -> None:
    story = await story_with(text("the first answer"), text("the second answer"))

    await story.type("a first thought")
    await story.type("/compact --text settled")
    await story.type("and another")

    first, second = story.requests()
    assert first.messages == (User(text="a first thought"),), "the first turn read the conversation as it stood"
    assert second.messages == (as_message(Summary(text="settled")), User(text="and another")), (
        "the next turn read the summary instead of the stretch it stands for"
    )


async def test_a_summary_named_at_the_tip_keeps_the_whole_prefix(story_with: Scripted) -> None:
    story = await story_with(text("the first answer"), text("the second answer"))

    await story.type("a first thought")
    await story.type("/sessions")
    listed = re.fullmatch(r"sessions: main \((e\d+)\)", story.screen()[-1])
    assert listed is not None, f"the listing is where a client learns an id to name: {story.screen()[-1]!r}"

    await story.type(f"/compact --begin-after {listed[1]} --text kept")
    await story.type("and another")

    _, second = story.requests()
    assert second.messages == (
        User(text="a first thought"),
        Assistant(text="the first answer"),
        as_message(Summary(text="kept")),
        User(text="and another"),
    ), "a summary that continues the tip keeps the prefix: what it takes off the chain is only what follows it"


async def test_the_plan_epoch_moves_once_per_compaction(story_with: Scripted) -> None:
    story = await story_with(text("one"))
    await story.type("a thought")
    tools = story.toolset_epoch

    assert story.plan_epoch == 0, "nothing has been compressed yet"
    await story.type("/compact --text first")
    await story.type("/compact --text second")

    assert story.plan_epoch == 2, "a compaction moves the plan once, and two compactions are two"
    assert story.toolset_epoch == tools, "a compaction says nothing about the tools, which is why they are two numbers"
