"""What a person is asked before a call changes something, and what remembering an answer means.

A posture is derived from what a tool declares, not chosen per call: writing a file is agreed to once per
path and then never again, replacing one is agreed to every time, deleting something permanently is
agreed to every time, and reading is not agreed to at all. These stories are about the person's side of
that — which question appears, how often, and what a no does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Result
from plyngent.kernel.trail import Consent
from tests.stories.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.trail import Entry
    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def _asked(story: Story) -> tuple[Entry, ...]:
    """Every question this conversation put to somebody, as the journal kept it."""
    rows = await story.host.trail.entries(story.session)
    return tuple(entry for entry in rows if isinstance(entry.payload, Consent))


async def test_a_write_asks_once_for_a_path_and_again_for_another(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_replace", {"path": "notes.txt", "old_string": "one", "new_string": "two"}),
        calling("edit_replace", {"path": "notes.txt", "old_string": "two", "new_string": "three"}),
        calling("write_file", {"path": "other.txt", "content": "fresh"}),
        text("done"),
    )
    _ = (story.workspace / "notes.txt").write_text("one", encoding="utf-8")

    await story.type("change some notes")
    await story.answer("y")
    await story.answer("y")

    assert [question.prompt for question in story.questions()] == [
        "allow edit_replace on notes.txt?",
        "allow write_file on other.txt?",
    ], "the second change to the same path was not asked about again, and a new path was"
    # The last call writes its file on a thread, so the story waits for it to have come back before
    # reading the disk: nothing here may turn "the work has not finished" into an assertion.
    await story.until("wrote 5 chars to other.txt")
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "three"
    assert (story.workspace / "other.txt").read_text(encoding="utf-8") == "fresh"


async def test_allowing_a_new_file_does_not_allow_editing_it(story_with: Scripted) -> None:
    """One path, two questions: creating a file is the lighter ask, and a yes answers only its own."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one\n"}),
        calling("edit_replace", {"path": "notes.txt", "old_string": "one", "new_string": "two"}),
        calling("edit_replace", {"path": "notes.txt", "old_string": "two", "new_string": "three"}),
        text("done"),
    )

    await story.type("write it, then change it twice")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    assert [question.prompt for question in story.questions()] == [
        "allow write_file on notes.txt?",
        "allow edit_replace on notes.txt?",
    ], "the second change is the same call as the first, and it is the first that had to be agreed to"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "three\n"


async def test_replacing_a_file_is_asked_about_every_time(story_with: Scripted) -> None:
    """An approval to create a file is not an approval to overwrite it: the two are different calls."""

    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one"}),
        calling("write_file", {"path": "notes.txt", "content": "two", "overwrite": True}),
        calling("write_file", {"path": "notes.txt", "content": "three", "overwrite": True}),
        text("done"),
    )

    await story.type("write some notes")
    for _ in range(3):
        await story.answer("y")

    assert [question.prompt for question in story.questions()] == ["allow write_file on notes.txt?"] * 3, (
        "the third write is the second replacement, and a replacement is not remembered"
    )
    await story.until("wrote 5 chars to notes.txt")
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "three"


async def test_every_question_put_to_a_person_is_one_row_of_the_journal(story_with: Scripted) -> None:
    """A yes and a no are both written down, and neither is written into anything a model reads.

    The record a call leaves says which approvals it *earned*, so a refusal earns nothing and cannot say
    the call was ever put to anybody. This is the other side of it: one row per question, at the level it
    was asked, with who answered and what they said — which is the only thing that can answer "which
    danger questions did a policy answer, and which did a person say no to".
    """
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        calling("delete_path", {"path": "notes.txt", "permanent": True, "intention": "take it back"}),
        text("done"),
    )
    await story.require("fs")

    await story.type("write it, then take it back")
    await story.answer("y")
    await story.answer("n")
    await story.finished()

    rows = await _asked(story)
    assert [(row.payload.tool, row.payload.call, row.payload.kind) for row in rows] == [
        ("write_file", "tool-1", "confirmation"),
        ("delete_path", "tool-2", "confirmation"),
    ], "which call asked, and which kind of question it was"
    assert [(row.payload.severity, row.payload.by, row.payload.accepted) for row in rows] == [
        ("routine", "human", True),
        ("danger", "human", False),
    ], "the level it was asked at, who answered, and what they said"
    assert [row.payload.question for row in rows] == [
        "allow write_file on notes.txt?",
        "allow delete_path on notes.txt?",
    ], "the question itself, not only the fact that one was asked"
    assert [row.begin_after for row in rows] == [None, None], "a question is a root: no turn continues from one"
    assert {row.id for row in rows} <= set((await story.host.trail.shape(story.session)).roots)
    assert not [said for request in story.requests() for said in request.messages if said.text.startswith("allow ")], (
        "a model is not told what a person agreed to before its call ran"
    )


async def test_a_refused_write_does_not_happen_and_the_model_is_told(story_with: Scripted) -> None:
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        text("as you wish"),
    )

    await story.type("write a note")
    await story.answer("n")

    assert not (story.workspace / "notes.txt").exists(), "a no means nothing was written"
    assert "write_file refused: the human did not allow this call" in story.screen()
    (_, second) = story.requests()
    refused = second.messages[-1]
    assert isinstance(refused, Result) and refused.text == "the human did not allow this call", (
        "the model is told what happened, not left guessing"
    )
    assert "hint:" not in refused.text, "and nothing is offered, because nothing lifts a person's no"


async def test_a_permanent_delete_asks_every_time(story_with: Scripted) -> None:
    story = await story_with(
        calling("delete_path", {"path": "gone.txt", "permanent": True, "intention": "remove it"}),
        calling("delete_path", {"path": "gone.txt", "permanent": True, "intention": "remove it again"}),
        text("done"),
    )
    _ = (story.workspace / "gone.txt").write_text("x", encoding="utf-8")
    await story.require("fs")

    await story.type("delete it")
    await story.answer("y")
    await story.answer("y")

    assert [question.prompt for question in story.questions()] == [
        "allow delete_path on gone.txt?",
        "allow delete_path on gone.txt?",
    ], "an irreversible call is not remembered, so the same question comes back"


async def test_reading_is_never_asked_about(story_with: Scripted) -> None:
    story = await story_with(calling("listdir", {"path": "."}), text("that is all"))
    await story.require("fs")

    await story.type("what is here?")
    await story.until("listdir ok")

    assert story.questions() == (), "a read changes nothing, so there is nothing to agree to"
