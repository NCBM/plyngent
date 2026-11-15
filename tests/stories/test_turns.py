"""A turn: what the model is asked, what it answers, what the calls did, and what stops it.

Every story here says what the model replies, because a turn is the one thing the kernel cannot do on
its own — and a reply it did not script is a failed story rather than an invented answer. What is
asserted is still only faces outside the boundary: the lines on the screen, what the model was asked, the
disk, and the exit code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.catalog import RESIDENT
from plyngent.kernel.model import Assistant, Result, User
from plyngent.wire.ask import Confirmation
from tests.stories.model import Held, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_a_typed_line_becomes_a_turn_and_the_answer_is_read(story_with: Scripted) -> None:
    story = await story_with(text("the first line is the title"))

    await story.type("look at the README")

    assert story.screen() == (
        "turn 1 started (chat)",
        "the first line is the title",
        "turn 1 finished",
    )
    (asked,) = story.requests()
    assert asked.messages == (User(text="look at the README"),), "the model saw the person's line"


async def test_the_model_is_given_the_conversation_so_far(story_with: Scripted) -> None:
    story = await story_with(text("one"), text("two"))

    await story.type("first")
    await story.type("second")

    first, second = story.requests()
    assert first.messages == (User(text="first"),)
    assert second.messages == (
        User(text="first"),
        Assistant(text="one"),
        User(text="second"),
    ), "the second turn continues the chain the first one left, and nothing had to be handed over"


async def test_the_model_is_offered_the_residents_and_told_what_is_missing(story_with: Scripted) -> None:
    story = await story_with(text("ok"))

    await story.type("what can you do?")

    (asked,) = story.requests()
    names = [tool.name for tool in asked.tools]
    assert names == sorted(names), "the block sits in front of the prompt, so its order is the name's"
    assert set(names) == {*RESIDENT, "ask_user_line"}, "what it always has, plus what the client offers"
    assert asked.system == (
        "6 more groups (15 tools): client:c1(1), document(1), fs(5), net(1), process(6), session(1) — "
        "use tool_require(group=…) to add one"
    ), "and one line says what it is not seeing, because a group nobody mentions cannot be asked for"


async def test_a_call_runs_and_the_model_reads_what_it_said(story_with: Scripted) -> None:
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("written"),
    )

    await story.type("write a note")
    await story.answer("y")

    assert story.screen() == (
        "turn 1 started (chat)",
        "calling write_file",
        "write_file ok: wrote 2 chars to notes.txt",
        "written",
        "turn 1 finished",
    )
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi", "the call did what it said"

    first, second = story.requests()
    assert first.messages == (User(text="write a note"),)
    person, answered, came_back = second.messages
    assert person == User(text="write a note")
    assert isinstance(answered, Assistant), "the answer that asked for the call is in the history as it was made"
    assert [call.tool for call in answered.calls] == ["write_file"], "calls and text travel in one message"
    assert isinstance(came_back, Result)
    assert (came_back.tool, came_back.text) == ("write_file", "wrote 2 chars to notes.txt"), (
        "the model is told what came of the call, which is what makes a second call possible"
    )


async def test_a_second_message_is_refused_while_a_turn_is_running(story_with: Scripted) -> None:
    held = Held(text("about time"))
    story = await story_with(held)

    await story.type("a first thought")
    await story.type("and another")

    assert story.screen() == (
        "turn 1 started (chat)",
        "turn refused: a turn is already running in this session (turn 1 is in the way)",
    ), "the refusal names the turn in the way, and no second turn started"

    held.ready.set()
    await story.settle()

    assert story.screen()[-2:] == ("about time", "turn 1 finished")


async def test_ctrl_c_stops_the_request_and_the_session_is_usable_again(story_with: Scripted) -> None:
    story = await story_with(Held(text("too late")), text("the next one"))

    await story.type("something slow")
    await story.press("ctrl-c")
    await story.type("try again")

    assert story.screen() == (
        "turn 1 started (chat)",
        "turn 1 finished: user",
        "turn 2 started (chat)",
        "the next one",
        "turn 2 finished",
    ), "the request was stopped, the turn said why, and the session took the next message"


async def test_a_denylisted_program_is_put_to_the_person_before_it_runs(story_with: Scripted) -> None:
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["rm", "notes.txt"]}], "intention": "clear the note"}),
        text("understood"),
    )

    await story.type("delete the note")
    await story.answer("n")

    (question,) = story.questions()
    assert isinstance(question, Confirmation)
    assert "'rm'" in question.prompt, "the question names the program it is about"
    assert question.scope.startswith("1/r0/tool-1"), "and it belongs to the call that asked it"
    assert story.screen()[1:3] == (
        "calling execute: clear the note",
        "execute refused: the command denylist refused 'rm'",
    ), "the model's own reason is shown, and the refusal is what came of the call"

    (_, second) = story.requests()
    refusal = second.messages[-1]
    assert isinstance(refusal, Result) and refusal.tool == "execute"
