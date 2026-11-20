"""A turn: what the model is asked, what it answers, what the calls did, and what stops it.

Every story here says what the model replies, because a turn is the one thing the kernel cannot do on
its own — and a reply it did not script fails the story, with no answer invented. What is
asserted is still only faces outside the boundary: the lines on the screen, what the model was asked, the
disk, and the exit code.
"""

from __future__ import annotations

import os
import re
import signal
from typing import TYPE_CHECKING

from plyngent.cli.policy import Policy
from plyngent.kernel.catalog import RESIDENT
from plyngent.kernel.failure import Failure
from plyngent.kernel.model import Assistant, Result, User
from plyngent.kernel.trail import Consent
from plyngent.wire.ask import Confirmation, Selection
from plyngent.wire.command import Open, Submit
from tests.stories.model import Failing, Held, calling, text

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
    assert set(names) == {*RESIDENT, "ask_user"}, "what it always has, plus what the client offers"
    assert asked.system.rsplit("\n\n", 1)[-1] == (
        "6 more groups (21 tools): document(1), fs(7), net(1), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one"
    ), "and the last section says what it is not seeing, because a group nobody mentions cannot be asked for"


async def test_a_call_runs_and_the_model_reads_what_it_said(story_with: Scripted) -> None:
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("written"),
    )

    await story.type("write a note")
    await story.answer("y")
    await story.finished()

    screen = story.screen()
    assert screen[:4] == (
        "turn 1 started (chat)",
        "calling write_file",
        "write_file ok: wrote 2 chars to notes.txt",
        "took 0 ms",
    ), "what the person sees first: the turn, the call, what came of it, and what it cost"
    assert re.fullmatch(r"write_file changed 1 path \([0-9a-f]{8}\)", screen[4]), (
        "then what the call left different — the version's own id, so a person can ask for it again"
    )
    assert screen[-2:] == ("written", "turn 1 finished"), "and the difference does not displace the answer"
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


async def test_a_person_can_ask_which_turn_is_working(story_with: Scripted) -> None:
    """The one fact between a turn's start and its end that a client cannot work out for itself.

    A turn begins because a command said so and ends because an event says so, and in between only the
    kernel knows: the word answers with the turn that is running, and with nothing once it is done — the
    answer the plain lane waits on before it leaves.
    """
    held = Held(text("about time"))
    story = await story_with(held)

    await story.type("a first thought")
    read_from = len(story.notices())
    await story.type("/running")
    await story.settle()

    assert story.notices()[read_from:] == ("turn 1 chat is running",), "the turn is named while it runs"

    held.ready.set()
    await story.finished()
    read_from = len(story.notices())
    await story.type("/running")
    await story.settle()

    assert story.notices()[read_from:] == ("nothing is running in this conversation",), (
        "and a conversation with nothing in flight says so, where silence would say nothing"
    )


async def test_an_aside_answers_beside_the_conversation_and_stays_out_of_it(story_with: Scripted) -> None:
    """``/btw`` costs one request and one answer, and the next turn is sent the conversation it had before.

    The rows are in the log — that is where an answer lives — and what makes them an aside is that the
    projection a model is given walks past them, until somebody says to keep them (E163).
    """
    story = await story_with(text("first answer"), text("second answer"))

    await story.type("/btw what about Tuesday")
    await story.answer("n")
    await story.until("turn 1 finished")
    await story.type("carry on")
    await story.finished(turn=2)

    _aside, next_turn = story.requests()
    assert [message.text for message in next_turn.messages] == ["carry on"], (
        "the aside is not part of the conversation the next request carries"
    )
    assert story.questions()[0].prompt == "keep this in the main conversation?"
    assert story.questions()[0].default is False, "and an empty line keeps it out, which is the quiet answer"


async def test_a_kept_aside_joins_the_conversation_from_then_on(story_with: Scripted) -> None:
    """A yes to that question writes one row, and a projection below it reads the aside's own rows."""
    story = await story_with(text("first answer"), text("second answer"))

    await story.type("/btw what about Tuesday")
    await story.answer("y")
    await story.until("turn 1 finished")
    await story.type("carry on")
    await story.finished(turn=2)

    _aside, next_turn = story.requests()
    assert [message.text for message in next_turn.messages] == ["what about Tuesday", "first answer", "carry on"]


async def test_an_aside_is_offered_only_what_writes_nothing(story_with: Scripted) -> None:
    """The tools of an aside are the ones no call could change anything with — a property, not a list."""
    story = await story_with(text("one"))

    await story.type("/btw what can you see")
    await story.answer("n")
    await story.finished()

    (aside,) = story.requests()
    offered = {tool.name for tool in aside.tools}
    assert "write_file" not in offered, "a tool that writes is not offered to a turn that changes nothing"
    assert "read" in offered, "and the ones that only read are"


async def test_a_signal_is_the_stop_a_key_is(story_with: Scripted) -> None:
    """A run reading a pipe has no key to press, so the same intention arrives as a signal and means the same.

    The signal is raised by the story at its own process, which is the only way to have one arrive: what
    is under test is that the loop answers it with a ``Cancel``, not by leaving, so the turn says
    why it stopped and the session takes the next message.
    """
    was = signal.signal(signal.SIGINT, lambda *_: None)
    try:
        story = await story_with(Held(text("too late")), text("the next one"))

        await story.type("something slow")
        _ = os.kill(os.getpid(), signal.SIGINT)
        await story.until("turn 1 finished: user")
        await story.type("try again")

        assert story.screen()[-2:] == ("the next one", "turn 2 finished"), (
            "the request was stopped by the signal, and the program is still here"
        )
    finally:
        _ = signal.signal(signal.SIGINT, was)


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
        "execute refused: the human did not allow running 'rm'",
    ), "the model's own reason is shown, and the refusal is what came of the call"
    (_, second) = story.requests()
    refusal = second.messages[-1]
    assert isinstance(refusal, Result) and refusal.tool == "execute"


async def test_a_retry_asks_again_and_repeats_no_call(story_with: Scripted) -> None:
    """``/retry`` is one more request against the chain, not a rerun of what the chain already holds (E158)."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one"}),
        Failing(Failure.TIMEOUT, "nobody answered"),
        text("asked again"),
    )
    await story.require("fs")

    await story.type("write it")
    await story.answer("y")
    await story.until("turn 1 failed")

    await story.type("/retry")
    await story.until("asked again")

    assert "turn 2 started (retry)" in story.screen(), "the turn kind is said, and it is nobody's chat"
    asked = story.requests()[-1]
    assert [type(message) for message in asked.messages] == [User, Assistant, Result], (
        "the retry adds no line of its own: it asks with the chain the failed turn left"
    )
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "one", (
        "and the call already made is not made again"
    )


async def test_a_turn_past_what_the_kernel_can_run_is_refused_and_not_queued(story_with: Scripted) -> None:
    """`[turn] running` is the process's budget, and being over it is an answer now, not later (E156)."""
    held = Held(text("the first answer"))
    story = await story_with(held, text("the second answer"), running=1)

    await story.type("one")
    assert story.client.submit(Open(session="other")), "the connection was already gone"
    await story.settle()
    assert story.client.submit(Submit(session="other", text="two")), "and so was the turn it asked for"
    await story.until("turn refused")

    assert "already running 1 turns" in "\n".join(story.screen()), (
        "the second conversation is told the ceiling, not left waiting behind it"
    )

    held.ready.set()
    await story.finished(1)
    assert story.client.submit(Submit(session="other", text="two, once there is room"))
    await story.until("the second answer")

    assert any("the second answer" in line for line in story.screen()), "and with a place free the same turn runs"


# -- the round wall -------------------------------------------------------------------------------


async def test_a_turn_stops_at_the_round_wall_and_says_both_numbers(story_with: Scripted) -> None:
    """`[turn] rounds` is a wall, and the sentence names how far the turn got and what the number was.

    Hard is the default, because an ordinary program that wrote a number down meant it: a model that keeps
    asking for one more tool is exactly the shape that would otherwise go round for ever, and nothing
    inside the loop could stop it (E150).
    """
    story = await story_with(
        calling("read", {"path": "notes.txt"}),
        calling("read", {"path": "notes.txt"}),
        rounds=2,
    )

    await story.type("keep looking")
    await story.until("turn 1 failed")

    failed = [line for line in story.screen() if line.startswith("turn 1 failed")]
    assert failed and "limit" in failed[0], f"the turn says where it stopped: {story.screen()!r}"
    assert "ran 2 rounds" in failed[0] and "2 rounds this program runs" in failed[0], (
        f"and both numbers are in that one sentence: {failed[0]!r}"
    )
    assert len(story.requests()) == 2, "nothing was asked past the number"


async def test_a_soft_wall_asks_and_an_answered_window_keeps_the_turn_going(story_with: Scripted) -> None:
    """`[turn] soft` turns the wall into a question, and the person's answer is the new window (E150).

    The rhythm is a window at a time: asked at two, allowed four, and the third
    round ended the work by itself so nothing was asked again.
    """
    story = await story_with(
        calling("read", {"path": "notes.txt"}),
        calling("read", {"path": "notes.txt"}),
        text("that is all"),
        rounds=2,
        soft=True,
    )

    await story.type("keep looking")
    await story.answer("another 2 rounds")
    await story.finished()

    (question,) = story.questions()
    assert isinstance(question, Selection), "three answers, so it is a choice, not a yes or a no"
    assert question.options == ("another 2 rounds", "no more limit for this turn", "stop here")
    assert question.default == "another 2 rounds", "and an empty line opens the window it would have anyway"
    assert len(story.requests()) == 3, "the turn went past the wall and finished on its own"

    rows = [row.payload for row in await story.host.trail.entries(story.session)]
    (consent,) = [row for row in rows if isinstance(row, Consent)]
    assert (consent.tool, consent.call, consent.kind, consent.by) == ("", "rounds", "selection", "human"), (
        "the record says the kernel asked this one itself, under the call the wall runs in"
    )


async def test_a_soft_wall_is_answered_by_the_policy_when_nobody_is_there(story_with: Scripted) -> None:
    """`--away` reaches the choice through its default, so an unattended run keeps the ceiling it was given.

    Nobody in front of the machine answered nothing: the option already in front of them was answered, and
    the screen says which policy did it — which is the whole difference between this and a wall.
    """
    story = await story_with(
        calling("read", {"path": "notes.txt"}),
        calling("read", {"path": "notes.txt"}),
        text("that is all"),
        rounds=2,
        soft=True,
        policy=Policy(away=True),
    )

    await story.type("keep looking")
    await story.finished()

    assert story.questions() == (), "the question never reached a person, because it had an answer in front of it"
    assert "away: allowed" in story.notices(), f"and the screen says which policy answered: {story.notices()!r}"
    assert len(story.requests()) == 3, "the window opened, and the turn finished inside it"


async def test_a_soft_wall_that_is_told_to_stop_ends_the_turn_there(story_with: Scripted) -> None:
    """The third answer is a stop, and it is not the same sentence as having nobody answer at all."""
    story = await story_with(
        calling("read", {"path": "notes.txt"}),
        calling("read", {"path": "notes.txt"}),
        rounds=2,
        soft=True,
    )

    await story.type("keep looking")
    await story.answer("stop here")
    await story.until("turn 1 failed")

    failed = [line for line in story.screen() if line.startswith("turn 1 failed")]
    assert failed and "stopped after 2 rounds" in failed[0], f"a person's stop, said as one: {failed[0]!r}"
    assert len(story.requests()) == 2, "and nothing was asked after it"


async def test_a_turn_with_no_rounds_in_the_file_is_never_counted(story_with: Scripted) -> None:
    """Zero is nobody having said a number, so a program nobody configured runs as long as the model does.

    Four rounds on one line and a fifth that ends the turn: the same shape as the wall story, with the
    wall taken away — which is what makes every story above a story about a number somebody wrote.
    """
    story = await story_with(
        *(calling("read", {"path": "notes.txt"}) for _ in range(4)),
        text("there it is"),
    )

    await story.type("keep looking")
    await story.finished()

    assert story.questions() == (), "and with no number there is nothing to ask about"
    assert len(story.requests()) == 5, "every round the model asked for happened"
