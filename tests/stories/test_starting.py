"""Starting the program, asking it what it can do, and leaving it again."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent_creative.model import Failing, text

from plyngent.kernel.failure import Failure

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_starting_says_nothing_and_costs_nothing(story: Story) -> None:
    assert story.screen() == (), "coming up is not something the kernel says"
    assert story.notices() == (), "and it does not ask the person anything either"


async def test_the_listing_is_the_kernels_own_table(story: Story) -> None:
    await story.type("/help")

    listing = story.notices()
    assert listing[0] == "commands you can type:"
    assert listing[-1] == "anything else you type is a message"
    for word in ("/open", "/close", "/cancel", "/list-sessions", "/submit"):
        assert any(line.strip().startswith(word) for line in listing), word
    assert "the same commands, shorter: /new = /open, /sessions = /list-sessions" in listing
    assert any(line.strip().startswith("/model [list [--persist] | use <id>]") for line in listing), (
        "and the word that chooses a model is not an alias of anything"
    )


async def test_one_command_can_describe_itself(story: Story) -> None:
    await story.type("/open --help")

    written = "\n".join(story.notices())
    assert written.startswith("Usage: open")
    assert "--session" in written, "the arguments come from the kernel's own schema"
    assert "Start a conversation" in written


async def test_a_word_nobody_has_is_a_complaint_and_not_a_message(story: Story) -> None:
    await story.type("/nope")

    assert story.screen() == (), "a complaint is not a kernel event, so stdout stays empty"
    assert story.notices() == ("no command /nope here; /help lists the ones there are",)


async def test_ctrl_d_leaves_with_zero(story: Story) -> None:
    await story.press("ctrl-d")

    assert await story.stop() == 0


async def test_the_kernel_leaving_ends_the_program_and_says_why(story: Story) -> None:
    story.connection.stop("going down")
    await story.settle()

    assert story.screen() == ("the kernel is leaving: going down",)
    assert await story.stop() == 0


async def test_one_message_is_a_whole_run(story_with: Scripted) -> None:
    """``-p`` is the same loop with a different goodbye: one message, one turn, and the code (E159)."""
    story = await story_with(text("an answer"), once="say something")

    assert story.questions() == (), "a message is not a question, and nothing had to be agreed to"
    assert await story.stop() == 0
    assert story.screen() == ("turn 1 started (chat)", "an answer", "turn 1 finished")


async def test_a_run_whose_turn_failed_leaves_with_one(story_with: Scripted) -> None:
    """The turn's failure is the run's error: a pipe has to be able to tell it from an answer (E159)."""
    story = await story_with(Failing(Failure.TIMEOUT, "nobody answered"))

    await story.type("a thought")
    await story.until("turn 1 failed")

    assert await story.stop() == 1
