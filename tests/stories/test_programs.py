"""One call that runs programs: what runs, what waits, and what the person and the model are told.

A command is the only tool whose work is a tree: a node may hand its output on, a non-zero exit may skip
what follows, and a call can stop half way with some of its commands already run. These stories are about
that tree from the outside — who is asked before a program starts, what the screen says came of the call,
and what the model is handed afterwards. Exit codes are the program's answer rather than the call's
verdict, so a story that wants a verdict has to look at the status line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


def _file(story: Story, name: str) -> None:
    """A file in the story's workspace, put there before the story says anything."""
    _ = (story.workspace / name).write_text("x", encoding="utf-8")


async def test_a_denylisted_program_is_asked_about_once_and_then_remembered(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {"steps": [{"argv": ["rm", "a.txt"]}, {"argv": ["rm", "b.txt"]}], "intention": "clear two notes"},
        ),
        calling("execute", {"steps": [{"argv": ["rm", "c.txt"]}], "intention": "and the last one"}),
        text("all gone"),
    )
    for name in ("a.txt", "b.txt", "c.txt"):
        _file(story, name)

    await story.type("delete the three notes")
    await story.answer("y")
    await story.finished()

    (_, first, second) = story.requests()
    assert [question.prompt for question in story.questions()] == ["run 'rm'? it is on the command denylist"], (
        "one program, one question: the yes is remembered for the host, not for the call"
    )
    assert told(first).count("cmd=") == 2, told(first)
    assert told(second).count("cmd=") == 1, "the second call ran its command without asking again"
    assert not any((story.workspace / name).exists() for name in ("a.txt", "b.txt", "c.txt"))


async def test_a_denied_program_stops_the_call_and_nothing_runs(story_with: Scripted) -> None:
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["rm", "keep.txt"]}], "intention": "remove the note"}),
        text("as you wish"),
    )
    _file(story, "keep.txt")

    await story.type("delete the note")
    await story.answer("n")
    await story.finished()

    (_, refused) = story.requests()
    assert told(refused) == "the command denylist refused 'rm'", told(refused)
    assert "execute refused: the command denylist refused 'rm'" in story.screen(), story.screen()
    assert (story.workspace / "keep.txt").exists(), "a no means the program never started"


async def test_a_branch_that_is_never_reached_is_never_asked_about(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [
                    {"argv": ["sh", "-c", "exit 1"], "then": [{"argv": ["rm", "x.txt"]}]},
                    {"argv": ["sh", "-c", "echo second"]},
                ],
                "failfast": False,
                "intention": "a branch that does not run",
            },
        ),
        text("done"),
    )
    _file(story, "x.txt")

    await story.type("run the tree")
    await story.finished()

    (_, ran) = story.requests()
    assert story.questions() == (), "the then-branch was skipped, so nobody was asked about its program"
    assert told(ran).count("cmd=") == 2, "both top-level commands ran, and the branch did not"
    assert (story.workspace / "x.txt").exists()


async def test_a_failure_with_nothing_after_it_is_still_a_result(story_with: Scripted) -> None:
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["sh", "-c", "exit 1"]}], "intention": "a program that fails"}),
        text("oh"),
    )

    await story.type("run something that fails")
    await story.finished()

    (_, ran) = story.requests()
    assert "exit_code=1" in told(ran), told(ran)
    assert "execute ok: 1 of 1 commands ran" in story.screen(), (
        "the call did what it was asked; the command is what failed"
    )


async def test_failfast_false_follows_the_failure_and_runs_the_next_node(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["sh", "-c", "exit 1"]}, {"argv": ["sh", "-c", "echo next"]}],
                "failfast": False,
                "intention": "keep going",
            },
        ),
        text("both ran"),
    )

    await story.type("run both")
    await story.finished()

    (_, ran) = story.requests()
    assert told(ran).count("cmd=") == 2, "the second command ran even though the first one failed"
    assert "next" in told(ran), told(ran)


async def test_what_ran_before_a_stop_is_partial_and_stays_run(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["sh", "-c", "echo first > first.txt"]}, {"argv": ["no-such-program"]}],
                "intention": "one command and one that is not there",
            },
        ),
        text("oh"),
    )

    await story.type("run a tree that stops")
    await story.finished()

    (_, ran) = story.requests()
    assert (story.workspace / "first.txt").exists(), "a command that ran stays run, whatever came after it"
    assert told(ran).count("cmd=") == 1, "the model is handed the command that ran, and not the one that did not"
    assert any(line.startswith("execute partial: 1 of 2 commands ran") for line in story.screen()), story.screen()


async def test_a_redirect_that_leaves_the_workspace_is_refused_before_it_runs(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {"steps": [{"argv": ["sh", "-c", "echo x"], "pipe": "../out.txt"}], "intention": "write next door"},
        ),
        text("oh"),
    )

    await story.type("write outside")
    await story.finished()

    (_, refused) = story.requests()
    assert "escapes workspace root" in told(refused), told(refused)
    assert not (story.workspace.parent / "out.txt").exists(), "it was refused before anything ran"
    assert story.questions() == (), "a directory it may not write is not a question for the person"


async def test_a_deadline_stops_the_command_and_says_it_did(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {"steps": [{"argv": ["sh", "-c", "sleep 5"], "timeout_secs": 0.05}], "intention": "wait too long"},
        ),
        text("it never answered"),
    )

    await story.type("wait for an answer that does not come")
    await story.finished()

    (_, ran) = story.requests()
    assert told(ran).startswith("1 of 1 commands ran: no answer within"), (
        "the model reads why it stopped, not only which command it ran"
    )
    assert "timed_out=true" in told(ran), told(ran)
    assert any(line.startswith("execute partial: 1 of 1 commands ran") for line in story.screen()), story.screen()


async def test_a_tree_that_could_not_run_is_an_argument_error_and_asks_nobody(story_with: Scripted) -> None:
    story = await story_with(
        calling("execute", {"steps": [], "intention": "run nothing"}),
        text("nothing to run"),
    )

    await story.type("run an empty tree")
    await story.finished()

    (_, refused) = story.requests()
    assert "must not be empty" in told(refused), told(refused)
    assert story.questions() == (), "an argument error is not something a person is asked about"
