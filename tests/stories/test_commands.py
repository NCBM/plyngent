"""Saying things: a typed line, a composed one, and the commands that are about sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

#: What the kernel says when there is no model to ask: nothing installed, or nothing chosen.
NOTHING_TO_ASK = "turn refused: this conversation has no model to ask: none is installed, or none was chosen"

if TYPE_CHECKING:
    from tests.stories.driver import Story


async def test_a_typed_line_is_refused_when_no_model_is_installed(story: Story) -> None:
    await story.type("look at the README")

    assert story.screen() == (NOTHING_TO_ASK,), "there is nothing to ask, and the kernel says so in its own words"


async def test_the_sessions_are_listed_with_the_one_in_focus(story: Story) -> None:
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",)


async def test_resume_says_what_it_is_missing(story: Story) -> None:
    """A command this build cannot carry out is answered, not swallowed: what a person typed it for is
    the one thing they need back.

    What it is missing is a *second* journal to point the conversation at. One journal per workspace means
    getting a conversation back is opening it, which is a different word and already exists.
    """
    await story.type("/resume")
    await story.type("/resume --store elsewhere")
    await story.type("/resume --session nowhere")

    assert story.screen() == (
        "resume refused: a conversation here is resumed by opening it: this workspace keeps one journal, "
        "and no second one",
        "resume refused: no store named 'elsewhere': this workspace keeps one journal, and nothing else to point at",
        "[nowhere] resume refused: no such session here: open one first",
    ), "each answer names the thing it lacks, and the last one is about the session, not the store"


async def test_a_new_session_is_admitted_and_listed(story: Story) -> None:
    await story.type("/new --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work",), "admission order, and no invention"


async def test_a_name_with_a_space_in_it_is_one_argument(story: Story) -> None:
    await story.type('/new --session "work two"')
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work two",), "the quotes are the line's, not the kernel's"


async def test_a_closed_conversation_nothing_was_said_in_leaves_the_list(story: Story) -> None:
    """What a listing answers is the journal's, so a conversation nothing was written in is closed *and*
    gone: closing stops this kernel serving it, and the journal never held it."""
    await story.type("/new --session work")
    await story.type("/sessions")
    await story.type("/close --session work")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main, work", "sessions: main")


async def test_a_conversation_can_be_named_and_keeps_its_id(story: Story) -> None:
    """/rename is a setting of the conversation: the listing shows the name beside the id, and clears it."""
    await story.type("/new --session work")
    await story.type("/rename --session work --name 'the parser fix'")
    await story.type("/sessions")
    await story.type("/rename --session work")
    await story.type("/sessions")

    assert story.screen() == (
        "[work] named the parser fix",
        "sessions: main, the parser fix [work]",
        "[work] name cleared",
        "sessions: main, work",
    ), "the name is what a person reads, and the id it answers to is still there"


async def test_a_deleted_conversation_leaves_the_list_and_keeps_its_rows(story: Story) -> None:
    """Only a session's own chain is append-only, so deleting is a row: the list changes, nothing is lost.

    Opening it by name still works, which is the point of keeping the rows: what was deleted is an entry
    in a list, not a conversation.
    """
    await story.type("/new --session work")
    await story.type("/delete --session work")
    await story.type("/sessions")
    await story.type("/open --session work")
    await story.type("/close --session work")
    await story.type("/delete --session nowhere")

    assert story.screen() == (
        "[work] removed from the list, and its rows kept",
        "sessions: main",
        "[nowhere] session refused: no such session here: open one first",
    ), "the list is what changed, and a conversation this kernel does not serve says so"


async def test_a_message_written_in_an_editor_is_sent_like_a_typed_line(story: Story) -> None:
    await story.type("/submit")
    await story.write("a longer thought,\nover two lines")

    assert story.screen() == (NOTHING_TO_ASK,)


async def test_submit_will_not_take_its_text_on_the_line(story: Story) -> None:
    await story.type("/submit now")

    assert story.notices() == ("/submit takes no argument: it opens an editor for the text",)
    assert story.screen() == ()


async def test_ctrl_c_stops_what_is_running_without_leaving(story: Story) -> None:
    await story.press("ctrl-c")
    await story.type("/sessions")

    assert story.screen() == ("sessions: main",), "the session is still there, and so is the program"
