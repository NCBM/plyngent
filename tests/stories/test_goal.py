"""A plan that is not finished, and the conversation that keeps going by itself.

The switch is a setting like the plan beside it: ``/goal on`` says that a turn ending with work left in
the stack is not the end of the work, so the kernel starts the next turn itself. What it must never become
is a loop, so the two stories that matter are the ones about *not* driving: a switch nobody threw, and a
turn that answered without doing anything (E146).

A goal turn is an ordinary turn in every other respect — its own number, its own end row, its own cost —
which is why what a story reads here is the turn list and the requests, and not a new vocabulary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Call, Reply, Result, User
from plyngent.wire.command import Submit
from plyngent.wire.event import TurnKind
from tests.stories.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


def _plan(*titles: str) -> Reply:
    """An answer that puts work on the stack and says nothing about it."""
    return calling("todo_push", {"titles": list(titles)})


async def test_a_plan_that_is_not_finished_keeps_the_conversation_going(story_with: Scripted) -> None:
    """The whole of the mode: a turn that left work in the plan is followed by one nobody asked for.

    Asserted on the requests, because that is what "kept going" means for a model: the second one carries
    the plan and the reason it is being asked, and the third one is the second's own doing rather than a
    line somebody typed at it. Every turn here is two rounds — a call, and the answer that ends it —
    because one reply is not one turn.
    """
    story = await story_with(
        _plan("write the parser"),
        text("the plan is written down"),
        calling("glob_paths", {"pattern": "*.py"}),
        text("nothing left to do"),
        text("waiting for you"),
    )

    await story.type("/goal on")
    await story.type("write the parser")
    await story.finished(turn=3)

    assert "turn 1 started (chat)" in story.screen()
    assert "turn 2 started (goal)" in story.screen(), "the second turn is the kernel's, and it says so"
    assert "turn 3 started (goal)" in story.screen()

    asked = story.requests()
    assert "Carrying on:" not in asked[0].system, "a turn somebody typed at is not a turn carrying on"
    carried = [one for one in asked if "Carrying on:" in one.system]
    assert len(carried) == 3, "every round of every turn the kernel started says why it exists"
    assert "1 item still open: write the parser" in carried[0].system, "and what is left of the plan"
    assert [message for message in carried[-1].messages if isinstance(message, User)] == [
        User(text="write the parser")
    ], "a goal turn writes no line of its own, so the conversation has the one the person said"


async def test_a_goal_turn_that_did_nothing_does_not_drive_another(story_with: Scripted) -> None:
    """A turn that only talked changed nothing, so asking again would be insisting rather than progress."""
    story = await story_with(_plan("write the parser"), text("the plan is written down"), text("nothing more to do"))

    await story.type("/goal on")
    await story.type("write the parser")
    await story.finished(turn=2)
    await story.settle()

    assert "turn 3 started" not in "\n".join(story.screen()), "the chain ends with the turn that acted on nothing"
    assert len(story.requests()) == 3, "one asking for the plan, one to end that turn, and the goal turn's own"


async def test_the_switch_is_off_until_a_person_turns_it_on(story_with: Scripted) -> None:
    """The default a conversation starts with: a plan is something the model keeps, not a clock."""
    story = await story_with(_plan("write the parser"), text("the plan is written down"))

    await story.type("write the parser")
    await story.finished(turn=1)
    await story.settle()

    assert len(story.requests()) == 2, "a turn that left work in the plan ends the turn all the same"
    assert "turn 2" not in "\n".join(story.screen())


async def test_the_switch_says_where_it_stands(story_with: Scripted) -> None:
    """``/goal`` with no word is the question, and the answer is the mode the session has."""
    story = await story_with()

    await story.type("/goal")
    assert "goal off: the plan waits" in story.screen(), story.screen()

    await story.type("/goal on")
    assert "goal on: an unfinished plan keeps this conversation going" in story.screen()

    await story.type("/goal")
    assert story.screen()[-2:] == (
        "goal on: an unfinished plan keeps this conversation going",
        "goal on: an unfinished plan keeps this conversation going",
    ), "asking twice says the same thing, and moves nothing"


async def test_a_client_cannot_start_a_goal_turn(story_with: Scripted) -> None:
    """``GOAL`` is the one kind nobody may ask for: it is this kernel saying the plan is not finished."""
    story = await story_with()

    assert story.client.submit(Submit(session=story.session, kind=TurnKind.GOAL, text="carry on"))
    await story.until("turn refused: a turn is started by a person or by an unfinished plan")

    assert story.requests() == (), "and nothing was asked of the model"


async def test_the_switch_is_still_on_after_a_restart(tmp_path, transport: str) -> None:
    """A setting like any other: it comes back out of the journal, so a new process does not ask again."""
    from tests.stories.driver import Story
    from tests.stories.model import ScriptedModel

    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    first = await Story.start(root, transport=transport, model=ScriptedModel([]), journal=journal)
    try:
        await first.type("/goal on")
        assert "goal on" in "\n".join(first.screen())
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(root, transport=transport, model=ScriptedModel([]), journal=journal)
    try:
        await second.type("/goal")
        assert "goal on: an unfinished plan keeps this conversation going" in second.screen()
    finally:
        await second.close()


async def test_a_goal_turn_reads_the_plan_it_inherited(story_with: Scripted) -> None:
    """The plan a goal turn works from is the stack, not a sentence somebody wrote: it is read back out.

    Two items, one of them done: what the prompt says is what is *open*, which is the one thing a model
    that has to choose a next step needs to be told.
    """
    story = await story_with(
        Reply(calls=(Call(id="c1", tool="todo_push", arguments='{"titles": ["write it", "test it"]}'),)),
        Reply(calls=(Call(id="c2", tool="todo_update", arguments='{"group": 0, "number": 1, "status": "done"}'),)),
        text("planned"),
        text("waiting for you"),
    )

    await story.type("/goal on")
    await story.type("start")
    await story.finished(turn=2)

    carried = [one for one in story.requests() if "Carrying on:" in one.system]
    assert carried, "the goal turn is the one that carries the plan"
    assert "1 item still open: test it" in carried[0].system, "the finished one is not work left to do"
    assert any(
        "updated group 0 item 1" in message.text for message in carried[0].messages if isinstance(message, Result)
    ), "and the request continues from what the calls came back with"
