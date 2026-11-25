"""A client that answers some of the kernel's questions itself, and says which ones it answered.

Both names here are flags, not settings, so these are stories about one run: what was answered
without a person, what was deliberately still put to them, and what the screen says afterwards. The
kernel half of the promise is that it cannot tell the difference — an answer is an answer with an
author, and every one of these carries ``policy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.cli.policy import Policy
from plyngent.kernel.trail import Consent
from plyngent_creative.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.trail import Consent as Row
    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def _answered(story: Story) -> tuple[Row, ...]:
    """Every question this conversation put, and who answered it, as the journal kept it."""
    rows = await story.host.trail.entries(story.session)
    return tuple(entry.payload for entry in rows if isinstance(entry.payload, Consent))


async def test_a_policy_that_says_yes_stops_at_the_question_nothing_undoes(story_with: Scripted) -> None:
    """Two levels answered without anybody, and the heaviest still put to the person."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        calling("edit_replace", {"path": "notes.txt", "old_string": "hi", "new_string": "hello"}),
        calling("delete_path", {"path": "notes.txt", "permanent": True, "intention": "take it back"}),
        text("done"),
        policy=Policy(yolo=True),
    )

    await story.require("fs")
    await story.type("write it, change it, then take it back")
    await story.answer("n")
    await story.finished()

    assert [question.prompt for question in story.questions()] == ["allow delete_path on notes.txt?"], (
        "the light and the ordinary question were answered; the one nothing here can undo was not"
    )
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hello"
    assert [line for line in story.notices() if line.startswith("yolo:")] == [
        "yolo: allowed",
        "yolo: allowed",
    ], "every question a policy answered leaves a line saying which policy and which way"
    assert [(row.tool, row.by, row.policy, row.accepted) for row in await _answered(story)] == [
        ("write_file", "policy", "yolo", True),
        ("edit_replace", "policy", "yolo", True),
        ("delete_path", "human", "", False),
    ], "and the journal says the same thing without the screen, policy name and all"


async def test_away_answers_each_question_as_its_own_silence_would_have(story_with: Scripted) -> None:
    """The light question is a yes while it waits and the ordinary one is not, so away says both at once."""
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "hi"}),
        calling("edit_replace", {"path": "notes.txt", "old_string": "hi", "new_string": "hello"}),
        text("done"),
        policy=Policy(away=True),
    )

    await story.type("write it, then change it")
    await story.finished()

    assert story.questions() == (), "nothing was put to somebody who said they were not there"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi", "and the ordinary one's no held"
    assert [line for line in story.notices() if line.startswith("away:")] == ["away: allowed", "away: refused"]
    # The count is written when the run stops, which is why the person here says goodbye and waits.
    await story.press("ctrl-d")
    await story.settle()
    assert story.notices()[-1] == "away: 2 answered (1 allowed, 1 refused)", (
        "the end of the stretch is one line, because forty answered questions are a shape and not a list"
    )


async def test_answering_the_heaviest_with_a_yes_is_a_switch_of_its_own(story_with: Scripted) -> None:
    """``--away-danger`` is what allows what nothing here can undo, and it says so once at the start."""
    story = await story_with(
        calling("delete_path", {"path": "gone.txt", "permanent": True, "intention": "clean up"}),
        text("done"),
        policy=Policy(away=True, away_danger=True),
    )
    _ = (story.workspace / "gone.txt").write_text("x", encoding="utf-8")
    await story.require("fs")

    await story.type("delete it")
    await story.finished()

    assert story.questions() == (), "the heaviest question was answered too — that is what the switch means"
    assert not (story.workspace / "gone.txt").exists()
    assert story.notices().count("away: the heaviest questions are answered with a yes too") == 1, (
        "said once, so that everything after it is readable"
    )
    assert "away: allowed" in story.notices()


async def test_away_without_the_second_switch_refuses_the_heaviest(story_with: Scripted) -> None:
    """The same story with one flag fewer: nothing is deleted, and the screen says who refused."""
    story = await story_with(
        calling("delete_path", {"path": "gone.txt", "permanent": True, "intention": "clean up"}),
        text("done"),
        policy=Policy(away=True),
    )
    _ = (story.workspace / "gone.txt").write_text("x", encoding="utf-8")
    await story.require("fs")

    await story.type("delete it")
    await story.finished()

    assert (story.workspace / "gone.txt").exists()
    assert "away: refused" in story.notices()
    assert "nobody answered in time" not in story.screen()[-1], "the policy answered; no deadline ran out"
