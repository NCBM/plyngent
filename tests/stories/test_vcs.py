"""A model asking a real checkout: five read-only calls, and what git's own words become.

The group is not resident, so each story asks for it first — that is the rule every group follows rather
than anything about version control. The repository is real (``tests.repositories``), because what these
stories are about is the translation: git's sentences and exit codes turned into a summary a model reads
first, a body it can read second, and the two endings that are not a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import repositories
from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request
    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

pytestmark = pytest.mark.skipif(repositories.MISSING, reason="the vcs group reads a repository by starting git")


def _names(request: Request) -> list[str]:
    """The tools one request offered, by name."""
    return [tool.name for tool in request.tools]


async def test_a_model_asks_what_the_checkout_is_and_what_changed(story_with: Scripted) -> None:
    """Three questions, one turn: which repository, what state it is in, and where the next commit lands."""
    story = await story_with(
        calling("vcs_kind", {"intention": "which version control this is"}),
        calling("vcs_status", {"intention": "what has changed since the last commit"}),
        calling("vcs_branch", {"intention": "where the next commit would go"}),
        text("that is the state of it"),
    )
    repositories.make(story.workspace)
    _ = (story.workspace / "readme.txt").write_text("hello\nthere\n", encoding="utf-8")
    branch = repositories.run(story.workspace, "branch", "--show-current").strip()

    await story.require("vcs")
    await story.type("what checkout is this, and what is in it?")
    await story.finished()

    first, kind, status, head = story.requests()
    assert {"vcs_kind", "vcs_status", "vcs_branch"} <= set(_names(first)), "the group arrived for this round"
    assert told(kind).startswith("git\nrepository: "), told(kind)
    assert "answers: status, diff, log, branch" in told(kind), "the one place a backend says what it can answer"
    assert told(status).startswith("1 path changed"), told(status)
    assert "readme.txt" in told(status), "and the body is git's own short status"
    assert told(head) == f"on {branch}\n{branch}", "the branch, as the answer and not as a sentence about it"
    assert "vcs_kind ok: git" in story.screen(), story.screen()
    assert "vcs_status ok: 1 path changed" in story.screen(), story.screen()


async def test_a_model_reads_a_diff_and_a_history(story_with: Scripted) -> None:
    """The two answers that are longer than a line: what changed, and what came before it."""
    story = await story_with(
        calling("vcs_diff", {"intention": "the change in the note"}),
        calling("vcs_log", {"intention": "what came before this"}),
        text("thanks"),
    )
    repositories.make(story.workspace)
    _ = (story.workspace / "readme.txt").write_text("hello\nagain\n", encoding="utf-8")

    await story.require("vcs")
    await story.type("what changed, and what came before it?")
    await story.finished()

    _, changed, history = story.requests()
    assert told(changed).startswith("1 file changed, +1 -0"), told(changed)
    assert "+again" in told(changed), "a unified diff, counted in the summary and shown in the body"
    assert told(history).startswith("1 commit"), told(history)
    assert "initial" in told(history), "the commit's own subject, dated the way short dates are"
    assert "vcs_diff ok: 1 file changed, +1 -0" in story.screen(), story.screen()


async def test_staged_asks_about_the_next_commit_rather_than_the_last_one(story_with: Scripted) -> None:
    """``staged`` is a question, so the same tool answers it — and the two answers are different diffs."""
    story = await story_with(
        calling("vcs_diff", {"staged": True, "intention": "what the next commit would record"}),
        calling("vcs_diff", {"intention": "what is not going into it"}),
        text("both, then"),
    )
    repositories.make(story.workspace)
    _ = (story.workspace / "readme.txt").write_text("hello\nstaged line\n", encoding="utf-8")
    repositories.run(story.workspace, "add", "readme.txt")
    _ = (story.workspace / "readme.txt").write_text("hello\nstaged line\nunstaged line\n", encoding="utf-8")

    await story.require("vcs")
    await story.type("what is staged, and what is not?")
    await story.finished()

    _, staged, unstaged = story.requests()
    assert "+staged line" in told(staged) and "+unstaged line" not in told(staged), told(staged)
    assert "+unstaged line" in told(unstaged) and "+staged line" not in told(unstaged), told(unstaged)


async def test_a_workspace_with_no_repository_is_an_answer_and_a_failure(story_with: Scripted) -> None:
    """One name says there is none; the other four say there is nothing for them to answer.

    The distinction is the point of ``vcs_kind`` existing at all: "this is not a checkout" is a fact about
    the workspace, and a ``vcs_status`` in one is a call with no subject.
    """
    story = await story_with(
        calling("vcs_kind", {"intention": "is this a checkout at all"}),
        calling("vcs_status", {"intention": "what changed, if anything"}),
        text("right"),
    )

    await story.require("vcs")
    await story.type("is this a checkout?")
    await story.finished()

    _, kind, status = story.requests()
    assert told(kind).startswith("none here\nno repository in or above "), told(kind)
    assert told(status) == "no repository in or above " + str(story.workspace), told(status)
    assert f"vcs_status failed: no repository in or above {story.workspace}" in story.screen(), story.screen()
