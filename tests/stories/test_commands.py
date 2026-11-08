"""Saying things: a typed line, a composed one, and the commands that are about sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.stories.driver import Story


async def test_typing_a_line_is_a_turn_the_kernel_cannot_run_yet(story: Story) -> None:
    await story.type("look at the README")

    assert story.screen() == ("turn refused: this kernel cannot run turns yet",), (
        "there is no turn runner, and the kernel says so in its own words"
    )


async def test_the_sessions_are_listed_with_the_one_in_focus(story: Story) -> None:
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",)


async def test_a_new_session_is_admitted_and_listed(story: Story) -> None:
    await story.type("/new --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work",), "admission order, and no invention"


async def test_a_closed_session_is_gone_from_the_list(story: Story) -> None:
    await story.type("/new --session work")
    await story.type("/sessions")
    await story.type("/close --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work", "sessions: main")


async def test_a_message_written_in_an_editor_is_sent_like_a_typed_line(story: Story) -> None:
    await story.type("/submit")
    await story.write("a longer thought,\nover two lines")

    assert story.screen() == ("turn refused: this kernel cannot run turns yet",)


async def test_submit_will_not_take_its_text_on_the_line(story: Story) -> None:
    await story.type("/submit now")

    assert story.notices() == ("/submit takes no argument: it opens an editor for the text",)
    assert story.screen() == ()


async def test_ctrl_c_stops_what_is_running_without_leaving(story: Story) -> None:
    await story.press("ctrl-c")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",), "the session is still there, and so is the program"
