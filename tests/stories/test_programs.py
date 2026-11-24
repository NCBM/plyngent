"""One call that runs programs: what runs, what waits, and what the person and the model are told.

A command is the only tool whose work is a tree: a node may hand its output on, a non-zero exit may skip
what follows, and a call can stop half way with some of its commands already run. These stories are about
that tree from the outside — who is asked before a program starts, what the screen says came of the call,
and what the model is handed afterwards. Exit codes are the program's answer, not the call's
verdict, so a story that wants a verdict has to look at the status line.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING

import pytest

from plyngent.cli.policy import Policy
from plyngent.kernel.trail import Consent
from plyngent.wire.ask import Severity
from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: What a story says when it is not about asking: yes to everything a program can raise, danger included.
#: A shell is a danger-level question here (§R12 乙), so a story that runs one says this, and does not
#: interrupt its own subject with an answer at the terminal.
_YES = Policy(away=True, away_danger=True)


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
    assert story.questions()[0].severity is Severity.DANGER, "and it is asked at the level a program is"
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
    assert told(refused) == "the human did not allow running 'rm'", told(refused)
    assert "execute refused: the human did not allow running 'rm'" in story.screen(), story.screen()
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
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    (_, ran) = story.requests()
    assert [question.prompt.split("?")[0] for question in story.questions()] == [
        "run sh -c 'exit 1'",
        "run sh -c 'echo second'",
    ], "one question per node that ran, and the skipped branch's program was never put to anybody"
    assert told(ran).count("cmd=") == 2, "both top-level commands ran, and the branch did not"
    assert (story.workspace / "x.txt").exists()


async def test_a_shell_is_asked_about_every_time_and_the_answer_is_not_remembered(story_with: Scripted) -> None:
    """A command this build cannot answer for is worth one call: there is no name to keep a yes by (§R12 乙)."""
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["sh", "-c", "echo one"]}], "intention": "first"}),
        calling("execute", {"steps": [{"argv": ["sh", "-c", "echo two"]}], "intention": "and again"}),
        text("both"),
    )

    await story.type("say one and then two")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    asked = [question.prompt for question in story.questions()]
    assert len(asked) == 2, "the second call is asked again, and does not trade on the first answer"
    assert all(prompt.startswith("run sh -c") for prompt in asked), asked
    assert all("runs code given on its own command line" in prompt for prompt in asked), "and says why"


async def test_the_code_a_shell_will_run_is_in_the_question_and_in_the_record(story_with: Scripted) -> None:
    """The line a shell is handed is the text that will run, so it is what the person reads and agrees to (E191)."""
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["sh", "-c", "echo one"]}], "intention": "say one"}),
        text("it said one"),
    )

    await story.type("say one")
    await story.answer("y")
    await story.finished()

    (question,) = story.questions()
    assert question.executable_code == "echo one", "the text that will run is the text that was shown"
    assert question.prompt.endswith("runs code given on its own command line, and its code is below")
    assert any("  echo one" in line for line in story.notices()), "and the block is on the screen under the sentence"
    rows = [row.payload for row in await story.host.trail.entries(story.session)]
    (consent,) = [row for row in rows if isinstance(row, Consent)]
    assert (consent.tool, consent.executable_code) == ("execute", "echo one"), (
        "the record says which program the person was shown, and not only that they said yes"
    )


async def test_a_script_the_call_will_read_is_the_code_in_the_question(story_with: Scripted) -> None:
    """An interpreter given a file reads it, so the question carries what the file says (E191)."""
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["sh", "say.sh"]}], "intention": "run the note"}),
        text("it said two"),
    )
    _ = (story.workspace / "say.sh").write_text("echo two\n", encoding="utf-8")

    await story.type("run the note")
    await story.answer("y")
    await story.finished()

    (question,) = story.questions()
    assert question.executable_code == "echo two\n", "a script is a file this call reads, and the person reads it too"


async def test_code_this_build_cannot_point_at_is_named_and_not_shown(story_with: Scripted) -> None:
    """``-m`` hands the interpreter a module: the code is on its own path, and the question says so."""
    if shutil.which("python3") is None:  # pragma: no cover — a module question needs a python to run
        pytest.skip("python3 is how a module is named")
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["python3", "-m", "json.tool"]}], "intention": "tidy the json"}),
        text("tidy"),
    )

    await story.type("tidy the json")
    await story.answer("y")
    await story.finished()

    (question,) = story.questions()
    assert question.executable_code == "", "there is no file to show, and an empty block would be a lie"
    assert question.prompt.endswith("its code is the module 'json.tool', which is not a file in this workspace")


async def test_a_call_can_take_a_name_out_of_the_environment(story_with: Scripted) -> None:
    """A command inherits what this program was started with, and inheriting is not always what was meant.

    ``GIT_DIR`` is the case this exists for: this program keeps its own versions by setting one, and a
    command that inherited it would be pointed at a repository nobody named. A null in the overlay is how
    a caller says "none of that here", which is the one thing no value can say — an empty one is a
    directory that does not exist.
    """
    was = os.environ.get("PLYNGENT_STORY_LEAK")
    os.environ["PLYNGENT_STORY_LEAK"] = "inherited"
    try:
        story = await story_with(
            calling(
                "execute",
                {
                    "steps": [
                        {
                            "argv": ["sh", "-c", "echo ${PLYNGENT_STORY_LEAK-absent}"],
                            "env": {"PLYNGENT_STORY_LEAK": None},
                        }
                    ],
                    "intention": "ask what the environment says",
                },
            ),
            text("nothing"),
            policy=_YES,
        )

        await story.type("run it")
        await story.finished()
    finally:
        if was is None:
            del os.environ["PLYNGENT_STORY_LEAK"]
        else:
            os.environ["PLYNGENT_STORY_LEAK"] = was

    (_, ran) = story.requests()
    assert "absent" in told(ran), "the name the caller took away is the one the command does not see"


async def test_a_failure_with_nothing_after_it_is_still_a_result(story_with: Scripted) -> None:
    story = await story_with(
        calling("execute", {"steps": [{"argv": ["sh", "-c", "exit 1"]}], "intention": "a program that fails"}),
        text("oh"),
        policy=_YES,
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
        policy=_YES,
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
        policy=_YES,
    )

    await story.type("run a tree that stops")
    await story.finished()

    (_, ran) = story.requests()
    assert (story.workspace / "first.txt").exists(), "a command that ran stays run, whatever came after it"
    assert told(ran).count("cmd=") == 1, "the model is handed the command that ran, and not the one that did not"
    assert any(line.startswith("execute partial: 1 of 2 commands ran") for line in story.screen()), story.screen()


async def test_a_pipe_carries_the_whole_file_and_says_how_much_crossed(story_with: Scripted) -> None:
    """`cat big.txt | grep -c`: the count is of the whole file, and the body says how many bytes went on.

    The first command's output is three times one body's cap. Read as its own output it would be cut;
    read as the next program's input it all arrives, so the count the model is handed is of every line.
    """
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["cat", "big.txt"], "pipe": {"argv": ["grep", "-c", "needle"]}}],
                "intention": "count the needles in the whole file",
            },
        ),
        text("counted"),
    )
    _ = (story.workspace / "big.txt").write_text("needle\n" * 20_000, encoding="utf-8")

    await story.type("count the needles")
    await story.finished()

    (_, ran) = story.requests()
    handoff = told(ran)
    assert "handed=140000 bytes" in handoff, handoff
    assert "20000" in handoff, "every line of the file reached `grep`, not the first 32 000 characters"
    assert "execute ok: 2 of 2 commands ran" in story.screen(), story.screen()


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
        policy=_YES,
    )

    await story.type("wait for an answer that does not come")
    await story.finished()

    (_, ran) = story.requests()
    assert told(ran).startswith("1 of 1 commands ran: no answer within"), (
        "the model reads why it stopped, not only which command it ran"
    )
    assert "timed_out=true" in told(ran), told(ran)
    assert any(line.startswith("execute partial: 1 of 1 commands ran") for line in story.screen()), story.screen()


async def test_a_process_that_outlived_its_group_is_said_to_the_model(story_with: Scripted) -> None:
    """E161: a deadline cannot reach what left the group, so the record says so and this end is closed.

    The command is a shell that starts a ``setsid`` in the background and then hangs. The kill takes the
    shell and everything still in its group; that one process is not in it and is still holding the
    output. What a model is handed is the newest truth there is — ``escaped=true`` beside ``timed_out=true``
    — because nothing portable can promise to stop that process, and leaving it unsaid is what makes a
    stray program a mystery later.
    """
    if shutil.which("setsid") is None:  # pragma: no cover — the case is util-linux's own program
        pytest.skip("setsid is how a process leaves the group this build kills")
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["sh", "-c", "setsid sleep 5 & echo out; sleep 5"], "timeout_secs": 0.05}],
                "intention": "leave something behind",
            },
        ),
        text("it left something behind"),
        policy=_YES,
    )

    await story.type("run something that escapes the group")
    await story.finished()

    (_, ran) = story.requests()
    assert "timed_out=true" in told(ran).splitlines(), told(ran)
    assert "escaped=true" in told(ran).splitlines(), told(ran)
    assert "out" in told(ran), "and what it said before the deadline is still the answer"


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
