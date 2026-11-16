"""The kernel asking the person something, and the answer finding its way back."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.wire.ask import By, Confirmation, Grades, Severity, Written
from tests.stories.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_a_question_is_asked_and_the_answer_goes_back(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.confirm(prompt="run the tests?"))
    await story.answer("y")

    assert (await asked).accepted is True
    question = story.questions()[0]
    assert isinstance(question, Confirmation), "a yes/no is a yes/no, not an envelope with switches"
    assert question.prompt == "run the tests?"
    assert story.notices() == ("run the tests? [y/n]",), "the person read the question, and nothing else"


async def test_a_suggested_answer_is_offered_and_a_blank_line_takes_it(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.confirm(prompt="run the tests?", default=True))
    await story.answer("")

    assert (await asked).accepted is True
    assert story.notices() == ("run the tests? [Y/n]",)


async def test_a_no_is_a_no(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.confirm(prompt="run the tests?"))
    await story.answer("n")

    assert (await asked).accepted is False


async def test_a_line_that_answers_nothing_is_asked_again(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.confirm(prompt="run the tests?"))
    await story.answer("maybe")
    await story.answer("y")

    assert (await asked).accepted is True
    assert story.notices() == ("run the tests? [y/n]", "run the tests? [y/n]"), "the same question, once more"


async def test_a_secret_is_taken_and_reported_nowhere(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.secret(prompt="the passphrase?"))
    await story.answer("hunter2")

    assert (await asked).text == "hunter2"
    written = "\n".join(story.screen() + story.notices())
    assert "hunter2" not in written, "a secret is only ever carried back, never shown"


async def test_a_question_is_refused_when_the_client_leaves_mid_answer(story: Story) -> None:
    asked = asyncio.ensure_future(story.human.confirm(prompt="run the tests?"))
    await story.settle()

    assert await story.stop() == 0
    assert (await asked).accepted is False, "the client is gone, so the question is refused at once"


async def test_nobody_is_asked_after_the_client_left(story: Story) -> None:
    assert await story.stop() == 0

    agreed = await story.human.confirm(prompt="anyone?")

    assert agreed.accepted is False, "no client is not no channel: the answer is no, not a hang"


async def test_a_question_that_runs_out_of_time_takes_the_prompt_with_it(story: Story) -> None:
    """Acceptance ①: a question asks for a hundredth of a second, and the hundredth passes."""
    asked = asyncio.ensure_future(story.human.text(prompt="what next?", timeout=0.01))

    answer = await asked
    await story.settle()

    assert answer.text is None, "nobody wrote anything, so the answer is the absence of one"
    assert story.questions()[0].timeout == 0.01, "the question says how long it is worth waiting for"
    assert story.editor.waiting == 0, "the prompt came down instead of standing there unread"


async def test_an_answer_that_arrives_too_late_changes_nothing(story: Story) -> None:
    """Acceptance ②: the wait is over, so an answer to it changes nothing.

    What a connection can say about a late answer is only that it was taken and handed over — whether
    anybody is still waiting is on the kernel's end of the lane, which over a socket is another process.
    So what is asserted is the consequence: nothing was answered, and nothing answers afterwards either.
    """
    asked = asyncio.ensure_future(story.human.text(prompt="what next?", timeout=0.01))
    await story.settle()
    question = story.questions()[0]

    assert (await asked).text is None
    assert story.kernel.client.answer(Written(id=question.id, text="late", by=By.HUMAN)) is True, (
        "the connection took it; whether that id still has a question is the kernel's answer"
    )
    await story.settle()
    assert (await asked).text is None, "and it answered nothing: the wait it belonged to was over"


async def test_a_question_with_no_deadline_of_its_own_gets_its_levels_grade(story_with: Scripted) -> None:
    """Acceptance ③: the level is where a question's deadline comes from, and the client counts to it."""
    story = await story_with(grades=Grades(confirm=0.01))

    asked = asyncio.ensure_future(story.human.text(prompt="what next?"))
    await story.settle()
    question = story.questions()[0]

    assert (await asked).text is None
    assert question.timeout == 0.01, "the level's grade, resolved before the question was sent"


async def test_a_level_with_no_grade_of_its_own_gets_the_connections_deadline(story_with: Scripted) -> None:
    """Acceptance ③ for the last resort: no number from the caller and none from the level either."""
    story = await story_with(deadline=0.01, grades=Grades(confirm=None))

    asked = asyncio.ensure_future(story.human.text(prompt="what next?"))
    await story.settle()
    question = story.questions()[0]

    assert (await asked).text is None
    assert question.timeout == 0.01, "the connection's own clock, and there is only one of them"


async def test_the_light_question_is_answered_by_its_own_silence(story_with: Scripted) -> None:
    """A ``routine`` question nobody answers is a yes, and the work happens without anybody there.

    This is what makes it true that a program can run unattended: the lightest level is a question, but
    not one worth stopping for.
    """
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("done"),
        grades=Grades(routine=0.01),
    )

    await story.type("write a note")
    await story.finished()

    question = story.questions()[0]
    assert question.severity is Severity.ROUTINE, "a new file in the workspace is the lightest ask"
    assert question.timeout_default is True, "and the question itself says what its silence means"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi", "silence was a yes"


async def test_the_ordinary_question_is_refused_by_its_own_silence(story_with: Scripted) -> None:
    """The level above it is not: nobody answering a ``confirm`` question means the call does not happen."""
    story = await story_with(
        calling("edit_replace", {"path": "notes.txt", "old_string": "one", "new_string": "two"}),
        text("done"),
        grades=Grades(confirm=0.01),
    )
    _ = (story.workspace / "notes.txt").write_text("one", encoding="utf-8")

    await story.type("change the note")
    await story.finished()

    question = story.questions()[0]
    assert question.severity is Severity.CONFIRM
    assert question.timeout_default is False, "the question says so before the deadline, not after"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "one", "silence is not consent"
    assert "edit_replace refused: nobody answered in time, so this call did not happen" in story.screen()


# -- waiting for a person, which is the same question with a deadline ------------------------------


async def test_a_wait_comes_back_with_what_the_person_wrote(story_with: Scripted) -> None:
    """Acceptance ③ of §0.11: the first of a wait's three endings — somebody wrote a line."""
    story = await story_with(
        calling("wait", {"seconds": 30, "intention": "they are reading the plan"}),
        text("thanks"),
    )

    await story.type("give them a moment")
    await story.answer("read it, go ahead")
    await story.finished()

    assert "wait ok: the person wrote: read it, go ahead" in story.screen()
    question = story.questions()[0]
    assert question.prompt.startswith("waiting up to 30s — "), "what they read says how long it lasts"
    assert "they are reading the plan" in question.prompt, "and why the model is waiting"


async def test_an_empty_line_wakes_a_wait(story_with: Scripted) -> None:
    """The second ending: an empty line is an answer, and it is not the same one as silence."""
    story = await story_with(calling("wait", {"seconds": 30, "intention": "wait for an answer"}), text("ah"))

    await story.type("wait for them to nod")
    await story.answer("")
    await story.finished()

    assert "wait ok: the person pressed enter on an empty line" in story.screen()


async def test_a_wait_that_runs_out_says_nobody_spoke(story_with: Scripted) -> None:
    """The third ending: the wait gives up on its own, and the turn goes on after it."""
    story = await story_with(
        calling("wait", {"seconds": 0.01, "intention": "see if they are there"}), text("all right")
    )

    await story.type("wait a moment")
    await story.finished()

    assert "wait ok: nobody spoke in 0.01s" in story.screen()


async def test_ctrl_c_stops_a_wait(story_with: Scripted) -> None:
    """A wait is a call like any other: the stop a person sends ends it, and the turn says why."""
    story = await story_with(calling("wait", {"seconds": 30, "intention": "wait for the build"}), text("stopped"))

    await story.type("wait for the build")
    await story.press("ctrl-c")
    await story.until("turn 1 finished: user")

    assert "calling wait: wait for the build" in story.screen(), "the call had started, and the sleep was its own"
