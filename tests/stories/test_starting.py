"""Starting the program, asking it what it can do, and leaving it again."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.stories.driver import Story


async def test_starting_says_nothing_and_costs_nothing(story: Story) -> None:
    assert story.screen() == (), "coming up is not something the kernel says"
    assert story.notices() == (), "and it does not ask the person anything either"


async def test_the_listing_is_the_kernels_own_table(story: Story) -> None:
    await story.type("/help")

    listing = story.notices()
    assert listing[0] == "commands:"
    assert listing[-1] == "anything else you type is a message"
    for word in ("/open", "/close", "/cancel", "/list-sessions", "/submit"):
        assert any(line.strip().startswith(word) for line in listing), word
    assert "aliases: /new = /open, /sessions = /list-sessions, /model = /set-route" in listing


async def test_one_command_can_describe_itself(story: Story) -> None:
    await story.type("/open --help")

    written = "\n".join(story.notices())
    assert written.startswith("Usage: open")
    assert "--session" in written, "the arguments come from the kernel's own schema"
    assert "Start serving a session" in written


async def test_a_word_nobody_has_is_a_complaint_and_not_a_message(story: Story) -> None:
    await story.type("/nope")

    assert story.screen() == (), "a complaint is not a kernel event, so stdout stays empty"
    assert story.notices() == ("no command /nope here; /help lists the ones there are",)


async def test_ctrl_d_leaves_with_zero(story: Story) -> None:
    await story.press("ctrl-d")

    assert await story.stop() == 0


async def test_the_kernel_leaving_ends_the_program_and_says_why(story: Story) -> None:
    await story.kernel.serving.stop("going down")
    await story.settle()

    assert story.screen() == ("the kernel is leaving: going down",)
    assert await story.stop() == 0
