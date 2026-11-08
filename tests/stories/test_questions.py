"""The kernel asking the person something, and the answer finding its way back."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.wire.ask import Confirmation

if TYPE_CHECKING:
    from tests.stories.driver import Story


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
