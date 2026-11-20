"""A version: what a call left, kept where a person can look, and shown to nobody else.

A change is the one thing in this kernel with a single reader. The model has its result — the tool's own
words — and is told nothing more; the person watching is sent the difference itself, whether or not a
model is even in the conversation. These stories are about that asymmetry, and about the silence a call
that changed nothing has to keep: nothing is said about a read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.model import Result
from tests import repositories
from tests.stories.driver import Story
from tests.stories.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    type Scripted = Callable[..., Awaitable[Story]]

pytestmark = pytest.mark.skipif(repositories.MISSING, reason="a version is kept by starting git")


async def test_a_change_is_shown_to_the_person_watching(story_with: Scripted) -> None:
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one\ntwo\n"}),
        text("written"),
    )

    await story.type("write the note")
    await story.answer("y")
    await story.finished()

    screen = story.screen()
    assert "write_file changed 1 path" in "\n".join(screen), "the person sees which call and how much"
    assert any(line == "+one" for line in screen), "and the difference itself, in git's own spelling"
    assert any(line.startswith("@@ ") for line in screen), "with the hunk header it is written in"


async def test_the_model_is_told_nothing_about_the_difference(story_with: Scripted) -> None:
    """Its result is the account of the call; a second account would be a second thing to keep in step."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one\n"}),
        text("written"),
    )

    await story.type("write the note")
    await story.answer("y")
    await story.finished()

    (_, second) = story.requests()
    came_back = second.messages[-1]
    assert isinstance(came_back, Result)
    assert "notes.txt" in came_back.text, "the tool's own line, which is what the model reads"
    assert "@@" not in came_back.text and "+++" not in came_back.text, "and no difference anywhere in it"


async def test_a_read_is_not_a_version(story_with: Scripted) -> None:
    """A store asked about every call would be a log of nothing happening."""
    story = await story_with(calling("read", {"path": "notes.txt"}), text("read it"))
    _ = (story.workspace / "notes.txt").write_text("one\n", encoding="utf-8")

    await story.type("read the note")
    await story.until("turn 1 finished")
    assert "changed" not in "\n".join(story.screen()), "a call that changed nothing is not spoken of"


async def test_a_rollback_puts_back_what_the_last_call_left(story_with: Scripted) -> None:
    """One version back is the content the version before it held — and it is itself a version."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "one\n"}),
        calling("write_file", {"path": "notes.txt", "content": "two\n", "overwrite": True}),
        text("written"),
        text("written again"),
    )

    await story.type("write the note, then write it again")
    await story.answer("y")
    await story.answer("y")
    await story.finished(1)
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "two\n"

    await story.type("/rollback")

    await story.until("rolled back to")
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "one\n", "the version before it"


async def test_a_rollback_leaves_a_file_somebody_wrote_over_alone(story_with: Scripted) -> None:
    """The bytes on disk are not what that version left, so it is refused, not quietly applied."""
    story = await story_with(calling("write_file", {"path": "notes.txt", "content": "one\n"}), text("written"))

    await story.type("write the note")
    await story.answer("y")
    await story.finished(1)
    _ = (story.workspace / "notes.txt").write_text("typed by a person\n", encoding="utf-8")

    await story.type("/rollback")

    await story.until("rollback refused")
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "typed by a person\n"
