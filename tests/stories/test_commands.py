"""Saying things: a typed line, a composed one, and the commands that are about sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tests.stories.driver import Story


async def test_a_typed_line_is_refused_when_no_model_is_installed(story: Story) -> None:
    await story.type("look at the README")

    assert story.screen() == ("turn refused: this kernel has no model installed, so it cannot run a turn",), (
        "there is nothing to ask, and the kernel says so in its own words"
    )


async def test_the_sessions_are_listed_with_the_one_in_focus(story: Story) -> None:
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",)


async def test_resume_says_what_it_is_missing(story: Story) -> None:
    """A command this build cannot carry out is answered, not swallowed: what a person typed it for is
    the one thing they need back."""
    await story.type("/resume")
    await story.type("/resume --store elsewhere")
    await story.type("/resume --session nowhere")

    assert story.screen() == (
        "resume refused: resuming needs a stored trajectory, and this build has no store to read one from",
        "resume refused: no stored trajectory named 'elsewhere': this build has no store to read one from",
        "[nowhere] resume refused: no such session here: open one first",
    ), "each answer names the thing it lacks, and the last one is about the session rather than the store"


async def test_a_new_session_is_admitted_and_listed(story: Story) -> None:
    await story.type("/new --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work",), "admission order, and no invention"


async def test_a_name_with_a_space_in_it_is_one_argument(story: Story) -> None:
    await story.type('/new --session "work two"')
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work two",), "the quotes are the line's, not the kernel's"


async def test_a_closed_session_is_gone_from_the_list(story: Story) -> None:
    await story.type("/new --session work")
    await story.type("/sessions")
    await story.type("/close --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work", "sessions: main")


async def test_a_message_written_in_an_editor_is_sent_like_a_typed_line(story: Story) -> None:
    await story.type("/submit")
    await story.write("a longer thought,\nover two lines")

    assert story.screen() == ("turn refused: this kernel has no model installed, so it cannot run a turn",)


async def test_submit_will_not_take_its_text_on_the_line(story: Story) -> None:
    await story.type("/submit now")

    assert story.notices() == ("/submit takes no argument: it opens an editor for the text",)
    assert story.screen() == ()


async def test_ctrl_c_stops_what_is_running_without_leaving(story: Story) -> None:
    await story.press("ctrl-c")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",), "the session is still there, and so is the program"
