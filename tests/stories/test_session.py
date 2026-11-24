"""What a conversation's plan is, and what the tools that keep it answer.

The stack is a LIFO of groups: one push is one level of breakdown, and a group's number is its depth,
so pushing a child breakdown never renames the one it came from. These stories are the two calls that
start a stack — and the window a listing opens on one.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from plyngent.kernel.model import Call, Note, Reply, Result
from plyngent.kernel.trail import NOTICE_MARKER
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request

    type Scripted = Callable[..., Awaitable[Story]]


def _batch(*calls: Call) -> Reply:
    """One answer that asks for several calls, in the order the model wrote them."""
    return Reply(calls=calls)


def _asked(tool: str, *, call: str = "", **arguments: object) -> Call:
    """One call of a batch: the model's own id for it, and its arguments as the model would spell them."""
    return Call(id=call or tool, tool=tool, arguments=json.dumps(arguments))


def _answers(request: Request) -> tuple[str, ...]:
    """Every result the model had been given by the end of one round, oldest first."""
    return tuple(message.text for message in request.messages if isinstance(message, Result))


async def _told(story: Story, request: int = -1) -> str:
    """Wait for the turn to end, then read the newest result the model had by one request."""
    await story.until("turn 1 finished")
    return _answers(story.requests()[request])[-1]


async def _round(story: Story, request: int, count: int) -> tuple[str, ...]:
    """Wait for the turn to end, then read what one whole round answered with."""
    await story.until("turn 1 finished")
    return _answers(story.requests()[request])[-count:]


async def test_a_push_past_the_stack_ceiling_is_refused(story_with: Scripted) -> None:
    """E187: a stack has as many levels as this program allows, and the refusal names both ways out.

    ``depth=2`` is the ceiling here, where a program nobody configured goes to three. What the number
    protects is the one thing a stack of groups is for: the TOP group is the work being done now, and a
    breakdown deeper than this has stopped naming any.
    """
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write it"]}),
        calling("todo_push", {"titles": ["test it"]}),
        text("that is as far as this goes"),
        depth=2,
    )

    await story.type("plan it")

    refused = await _told(story)
    assert refused.startswith("the stack already has 2 groups, which is as deep as this program goes"), refused
    assert "hint: todo_update — work the TOP group" in refused, "one way out is to do the work that is there"
    assert "hint: todo_pop — or finish that level and pop it" in refused, "the other is to close the level"


async def test_a_pushed_group_is_listed_under_its_own_number(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["write the parser", "test it"]}),
        calling("todo_list", {}),
        text("that is the plan"),
    )

    await story.type("here is the plan")

    assert await _told(story, 1) == (
        "pushed 2 items (1 group)\nGroup 0 (TOP):\n  [ ] 1. write the parser\n  [ ] 2. test it"
    ), "a push answers the group it pushed, under the number the listing will use for it"
    assert await _told(story) == (
        "1 group, 2 items (2 open)\nGroup 0 (TOP):\n  [ ] 1. write the parser\n  [ ] 2. test it"
    ), "and the listing prints the same two tasks, with the marks they are at"
    assert "todo_push ok: pushed 2 items (1 group)" in story.screen(), "the person sees the same count"


async def test_pushing_a_breakdown_leaves_the_group_under_it_numbered_the_same(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write the parser"]}),
        calling("todo_list", {"full": True}),
        text("good"),
    )

    await story.type("break it down")

    assert await _told(story, 2) == (
        "pushed 1 item (2 groups)\n"
        "Group 1 (TOP):\n"
        "  [ ] 1. write the parser\n"
        "hint: todo_list(full=true) — 1 more group not shown"
    ), "a change answers the layer it changed, and says the rest of the stack was not drawn"
    assert await _told(story) == (
        "2 groups, 2 items (2 open)\nGroup 0:\n  [ ] 1. read the spec\nGroup 1 (TOP):\n  [ ] 1. write the parser"
    ), "the group that was already there kept its number, which is the whole point of depth numbering"


async def test_two_pushes_in_one_answer_are_two_groups(story_with: Scripted) -> None:
    story = await story_with(
        _batch(
            _asked("todo_push", call="first", titles=["alpha"]),
            _asked("todo_push", call="second", titles=["beta"]),
        ),
        calling("todo_list", {"full": True}),
        text("both noted"),
    )

    await story.type("two separate things to do")

    assert await _round(story, 1, 2) == (
        "pushed 1 item (1 group)\nGroup 0 (TOP):\n  [ ] 1. alpha",
        "pushed 1 item (2 groups)\nGroup 1 (TOP):\n  [ ] 1. beta\nhint: todo_list(full=true) — 1 more group not shown",
    ), "a batch really does run at once, so the second push reads the stack the first one wrote"
    assert await _told(story) == (
        "2 groups, 2 items (2 open)\nGroup 0:\n  [ ] 1. alpha\nGroup 1 (TOP):\n  [ ] 1. beta"
    ), "and neither push was lost"


async def test_a_plan_is_still_there_after_a_restart(tmp_path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("todo_push", {"titles": ["write the parser"]}), text("noted")]),
        journal=journal,
    )
    try:
        await first.type("here is the plan")
        assert "todo_push ok: pushed 1 item (1 group)" in first.screen(), first.screen()
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("todo_list", {}), text("still here")]),
        journal=journal,
    )
    try:
        await second.type("what is the plan?")
        await second.finished(turn=2)

        assert _answers(second.requests()[-1])[-1] == (
            "1 group, 1 item (1 open)\nGroup 0 (TOP):\n  [ ] 1. write the parser"
        ), "the stack is the conversation's, so a new process reads it back out of the history"
    finally:
        await second.close()


async def test_a_listing_shows_the_top_group_and_says_what_it_left(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write the parser"]}),
        calling("todo_list", {}),
        calling("todo_list", {"full": True}),
        text("there it is"),
    )

    await story.type("what am I doing?")

    assert await _told(story, 3) == (
        "1 of 2 groups, 2 items (2 open)\n"
        "Group 1 (TOP):\n"
        "  [ ] 1. write the parser\n"
        "hint: todo_list(full=true) — 1 more group not shown"
    ), "the window is the top group, and what it left out is offered as the call that would show it"
    assert await _told(story) == (
        "2 groups, 2 items (2 open)\nGroup 0:\n  [ ] 1. read the spec\nGroup 1 (TOP):\n  [ ] 1. write the parser"
    ), "full opens the window on the whole stack, bottom first"


async def test_a_push_of_nothing_but_blank_titles_is_the_arguments(story_with: Scripted) -> None:
    story = await story_with(calling("todo_push", {"titles": ["  ", "\t"]}), text("right"))

    await story.type("push nothing")

    assert await _told(story) == (
        "titles: give at least one title: a group whose every task is blank is not a level of breakdown"
    ), "a level of breakdown with no tasks in it is not a plan, so the call is refused before it runs"
    assert "todo_push failed: titles:" in "\n".join(story.screen()), "and the person sees which field it was"


async def test_a_task_is_changed_by_its_two_numbers(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write the parser"]}),
        calling("todo_push", {"titles": ["ship it"]}),
        calling(
            "todo_update",
            {"group": 1, "number": 1, "status": "in_progress", "title": "write the parser", "notes": "after the spec"},
        ),
        calling("todo_list", {"full": True}),
        text("noted"),
    )

    await story.type("start on the parser")

    assert await _told(story, 4) == (
        "updated group 1 item 1 → in_progress (title, notes)\n"
        "Group 1:\n"
        "  [~] 1. write the parser — after the spec\n"
        "hint: todo_list(full=true) — 2 more groups not shown"
    ), "a middle layer is answered without (TOP), and nothing either side of it is drawn"
    assert await _told(story) == (
        "3 groups, 3 items (3 open)\n"
        "Group 0:\n"
        "  [ ] 1. read the spec\n"
        "Group 1:\n"
        "  [~] 1. write the parser — after the spec\n"
        "Group 2 (TOP):\n"
        "  [ ] 1. ship it"
    ), "and the task it named is the one that changed, with its status, its title and its note"


async def test_a_group_number_the_stack_does_not_have_is_refused(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_update", {"group": 3, "number": 1, "status": "done"}),
        calling("todo_list", {"full": True}),
        text("right"),
    )

    await story.type("finish the third one")

    assert await _told(story, 2) == (
        "no group 3: the stack has 1 (0 is the bottom)\nhint: todo_list — both numbers come from the latest listing"
    ), "the answer says what numbers there are, and points at the listing that prints them"
    assert await _told(story) == ("1 group, 1 item (1 open)\nGroup 0 (TOP):\n  [ ] 1. read the spec"), (
        "and nothing was changed by a call whose numbers named nothing"
    )


async def test_a_task_number_the_group_does_not_have_is_refused(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec", "write the parser"]}),
        calling("todo_update", {"group": 0, "number": 3, "status": "done"}),
        calling("todo_list", {"full": True}),
        text("right"),
    )

    await story.type("finish the third one")

    assert await _told(story, 2) == (
        "no item 3 in group 0: it has 2 items\nhint: todo_list — both numbers come from the latest listing"
    ), "the item number is judged inside its group, and the group is named in the answer"
    assert await _told(story) == (
        "1 group, 2 items (2 open)\nGroup 0 (TOP):\n  [ ] 1. read the spec\n  [ ] 2. write the parser"
    ), "and nothing was changed"


async def test_a_group_number_below_zero_never_reaches_the_stack(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_update", {"group": -1, "number": 1, "status": "done"}),
        calling("todo_list", {"full": True}),
        text("right"),
    )

    await story.type("finish the last one")

    assert "group" in await _told(story, 2), (
        "a negative depth is not a number the listing could have printed, so the schema refuses it"
    )
    assert "todo_update failed: " in "\n".join(story.screen()), "and the person sees the call did not run"
    assert await _told(story) == ("1 group, 1 item (1 open)\nGroup 0 (TOP):\n  [ ] 1. read the spec"), (
        "nothing was changed"
    )


async def test_a_pop_takes_the_whole_top_group_and_answers_the_one_under_it(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write the parser", "test it"]}),
        calling("todo_pop", {}),
        calling("todo_list", {"full": True}),
        text("that level is done"),
    )

    await story.type("that breakdown is not the question")

    assert await _told(story, 3) == ("popped 2 items (1 group)\nGroup 0 (TOP):\n  [ ] 1. read the spec"), (
        "a pop takes the level off whole, and answers the group that is now the top one"
    )
    assert await _told(story) == ("1 group, 1 item (1 open)\nGroup 0 (TOP):\n  [ ] 1. read the spec"), (
        "the tasks of the group that went are gone with it, not left behind as finished ones"
    )


async def test_a_pop_of_an_empty_stack_says_where_to_look(story_with: Scripted) -> None:
    story = await story_with(calling("todo_pop", {}), text("fair enough"))

    await story.type("nothing to do then")

    assert await _told(story) == ("nothing to pop: the stack is empty\nhint: todo_list — see what is on the stack"), (
        "popping nothing is not a failure of the world; it is a stack with nothing on it"
    )


async def test_clearing_the_stack_says_how_much_went(story_with: Scripted) -> None:
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_push", {"titles": ["write the parser", "test it"]}),
        calling("todo_clear", {}),
        text("clean slate"),
    )
    await story.require("session")

    await story.type("forget all of it")

    assert await _told(story, 3) == ("cleared 3 items\n(todo stack empty — no groups)"), (
        "how much went is counted over the whole stack, and what is left is said once"
    )


async def test_clearing_the_stack_needs_the_group_first(story_with: Scripted) -> None:
    """The one place §10.4's thirteen names cost something: ``todo_clear`` arrives with its group."""
    story = await story_with(calling("todo_clear", {}), text("right"))

    await story.type("forget all of it")

    assert await _told(story) == (
        "'todo_clear' is not available yet: it is in the 'session' group\n"
        "hint: tool_require(group=session) — show that group first"
    ), "a session that has not asked for the group is answered with the one call that shows it"


async def test_what_is_left_of_the_plan_is_said_above_the_next_answer(story_with: Scripted) -> None:
    """E142: the plan lives in the session, so the model is told what is left of it while it works.

    A checklist written two rounds ago is one a model goes on working from memory, and a notice is the
    one row that fixes that: the round after the call carries it as the harness speaking — marker and all
    — above that round's own line, and the person watching is told where it was written.
    """
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec", "write the parser"]}),
        text("that is the plan"),
    )

    await story.type("here is the plan")
    await story.until("turn 1 finished")

    _first, second = story.requests()
    assert isinstance(second.messages[-1], Note), "the reminder is the last thing the model reads"
    assert second.messages[-1].text == (f"{NOTICE_MARKER}\n2 items still open: read the spec, write the parser"), (
        "the marker is in the text, so the model can tell the harness from the person"
    )
    assert isinstance(second.messages[-2], Result), "and it sits above the result it was written after"
    assert "written: 2 items still open: read the spec, write the parser" in "\n".join(story.screen()), (
        "the person reads the same sentence, under the entry it was written at"
    )


async def test_a_plan_that_was_cleared_says_the_reminder_no_longer_holds(story_with: Scripted) -> None:
    """A reminder that stopped being true is a row with no text: the log has it, the model does not.

    That is the difference between "the plan changed again" and "there is no plan any more": the second
    has nothing to say to the model, and a person reading the history still needs to know it happened.
    """
    story = await story_with(
        calling("todo_push", {"titles": ["read the spec"]}),
        calling("todo_clear", {}),
        text("clean slate"),
    )
    await story.require("session")

    await story.type("plan it, then drop it")
    await story.until("turn 1 finished")

    told = [message for message in story.requests()[-1].messages if isinstance(message, Note)]
    assert told == [Note(text=f"{NOTICE_MARKER}\n1 item still open: read the spec")], (
        "the reminder that held is the one the model was given, and the empty one is not a message"
    )
    assert "written: nothing is left on the plan" in "\n".join(story.screen()), "while a person reads what went away"
