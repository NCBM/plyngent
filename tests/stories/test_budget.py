"""A conversation that stops fitting the route it is on: what is put aside, and what is said about it.

A budget is invisible until it acts, and when it acts it does things a person can see: it writes a
checkpoint — a prune row, the text it kept, and the line the model reads about both — and, when nothing can
be put aside, it ends the turn at the wall instead of sending a request a provider would reject. Both are
stories, because both are things this program does to a conversation somebody is having (E149).

Every window here is a real one, and it has to be: this program's standing prompt — a persona, the
playbook and the tool declarations — is nearly six thousand tokens. The numbers below are chosen
against that, and a turn that ends at the wall says so in its own line, so the arithmetic is never a
guess a person has to make.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.lm.caps import Caps
from plyngent.kernel.lm.route import Route
from plyngent.kernel.model import Note, Reply, User
from plyngent.kernel.model import Usage as Cost
from plyngent.kernel.trail import NOTICE_MARKER
from tests.stories.model import text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: The checkpoint on the screen: the row, and how many entries it put aside.
WRITTEN = re.compile(r"prune e\d+ written \((\d+) withheld\)")

#: And the line the model is owed about it, said at the boundary where it can be read.
NOTICED = re.compile(r"notice e\d+ written: \d+ earlier entries put aside to fit the window")

#: The three lines the model reads instead of the stretch. The handle's id is the store's to make, so it
#: is matched rather than written down; the wording around it is model-visible and pinned.
HIDDEN = (
    "Hidden from you: 2 earlier entries of this conversation, put aside because it outgrew the room this "
    "route gives it. They are kept, not gone:"
)
KEPT = re.compile(r"\[d\d+\] 2 earlier entries put aside \(\d+ chars\)")
TAIL = (
    "Read that with `read(document_id=…)` when it matters. What you read below is not the whole "
    "conversation, so do not assume that a decision or a result you cannot see did not happen."
)

#: An answer long enough that keeping it would not leave room for another one: twenty thousand characters
#: is five thousand tokens, which is most of a seven thousand token window.
LONG = "x" * 20_000

#: An answer no summary of it could be sent whole: thirty thousand characters is seven and a half thousand
#: tokens, which is more than the room a compaction has once its own instruction is counted.
HUGE = "y" * 30_000


def _route(name: str, *, window: int) -> Route:
    """One endpoint that states what it can hold. Nothing answers at it: the scripted model does."""
    return Route(
        name=name,
        dialect="messages",
        base=f"http://{name}.invalid/v1",
        model="m",
        caps=Caps(max_output=64, context_window=window),
    )


async def test_a_conversation_that_outgrows_its_window_puts_the_oldest_rounds_aside(story_with: Scripted) -> None:
    """It is written down rather than trimmed away: a row, a line about it, and the entries still there."""
    story = await story_with(text(LONG), text("done"), routes=(_route("roomy", window=7_000),), route="roomy")

    await story.type("a first question")
    await story.finished(1)
    await story.type("a second question")
    await story.finished(2)

    assert story.requests()[0].messages == (User(text="a first question"),), (
        "the first request carried the whole conversation, because it fitted"
    )
    assert any(WRITTEN.fullmatch(line) for line in story.screen()), (
        f"a checkpoint says what it put aside: {story.screen()!r}"
    )
    assert any(NOTICED.fullmatch(line) for line in story.screen()), (
        f"and the model is owed a line about it: {story.screen()!r}"
    )
    said = story.requests()[-1].messages
    assert said[0] == User(text="a second question"), (
        "the round that no longer fitted is gone from what the model reads"
    )
    assert isinstance(said[1], Note), f"and the hole it left is named: {said!r}"
    assert said[1].text.split("\n")[0] == NOTICE_MARKER
    assert said[1].text.split("\n")[1] == HIDDEN, said[1].text
    assert KEPT.fullmatch(said[1].text.split("\n")[2]) is not None, (
        f"what was put aside is one document, and the line names it: {said[1].text!r}"
    )
    assert said[1].text.split("\n")[3] == TAIL, "and says the call that reads it back"


async def test_a_route_that_counts_three_times_as_much_is_measured_at_what_it_charged(story_with: Scripted) -> None:
    """The characters were only ever a guess, and the provider's own count of the same payload is the fix.

    Every number here is chosen so the two runs differ: this build reads the first request as ~5.8k tokens,
    the route says it charged 17.4k for exactly that, and from then on it is counted at three times what
    the text looks like. The window is wide enough for the conversation as this build reads it *and* narrow
    enough that the corrected reading does not fit — so a checkpoint on the second turn is the correction
    being in force, and no checkpoint is it having been written down and never used.
    """
    story = await story_with(
        Reply(text=LONG, usage=Cost(prompt=17_400)),
        text("done"),
        routes=(_route("roomy", window=20_000),),
        route="roomy",
    )

    await story.type("a first question")
    await story.finished(1)
    await story.type("a second question")
    await story.finished(2)

    assert story.requests()[0].messages == (User(text="a first question"),), (
        "nothing had been measured yet, so the first request carried the whole conversation"
    )
    said = story.requests()[-1].messages
    assert said[0] == User(text="a second question"), (
        f"and the second one did not fit what this route actually counts: {said!r}"
    )
    assert isinstance(said[1], Note), "which is a checkpoint, and the model is told about it"


async def test_a_summary_of_a_conversation_that_does_not_fit_leaves_the_beginning_out(story_with: Scripted) -> None:
    """A compaction is the one request that cannot be checkpointed to make room for itself.

    So its input is cut instead — from the oldest, since a summary of the part somebody is working on is
    still a summary — and the instruction says so, because a summary written as if the whole conversation
    had been read is worse than a short one.
    """
    story = await story_with(
        text(HUGE), text("a summary of it"), routes=(_route("roomy", window=7_000),), route="roomy"
    )

    await story.type("a first question")
    await story.finished()
    await story.type("/compact")
    await story.until("summary e")

    asked = story.requests()[-1]
    assert asked.system.startswith("You are a professional coding agent"), "a compaction keeps who is writing"
    assert not asked.tools, "and carries no tools: nothing it says is going to be used"
    assert [message.text for message in asked.messages[:-1]] == [], (
        f"nothing of the conversation fitted, so nothing of it was sent: {asked.messages!r}"
    )
    instruction = asked.messages[-1].text
    assert instruction.startswith("2 entries of this conversation did not fit what this route can hold"), instruction
    assert "Write a summary of the conversation above" in instruction, "and the plan's own instruction still ends it"


async def test_a_conversation_that_cannot_fit_at_all_ends_at_the_wall(story_with: Scripted) -> None:
    """The floor: a window that cannot hold this program's own prompt is not a conversation to prune.

    Pruning to nothing would be a request carrying an instruction and no history, which is not what the
    person asked for; sending it would be a provider saying no. Saying so is the third answer.
    """
    story = await story_with(text("hello"), routes=(_route("narrow", window=1_000),), route="narrow")

    await story.type("hi")
    await story.until("turn 1 failed")

    failed = [line for line in story.screen() if line.startswith("turn 1 failed")]
    assert failed and "limit" in failed[0], f"the turn says where it stopped: {story.screen()!r}"
    assert "needs" in failed[0] and "holds" in failed[0], "and how big it was against how big the room is"
    assert story.requests() == (), "nothing was sent: a request that cannot be read back is not worth making"
