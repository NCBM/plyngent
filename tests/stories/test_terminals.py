"""A program on a terminal: what it is told, what it answers, and who is asked before it is touched.

The terminal is the one tool whose answer carries a value the call after it has to carry: the session the
kernel hands out. A story cannot write that number down before the run, so these stories take it out of
the last thing the model was read — which is the other half of a scripted model, a step decided by the
request it answers (:data:`~tests.stories.model.Step`).

The children here are a shell that reads one line and repeats it, and ``cat`` for the terminal that has
to stay alive while something is written into it. Every story ends with the kernel going down, and that
is what closes the terminals a story left running.
"""

from __future__ import annotations

import re
import shlex
import sys
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.pty.registry import Limits
from plyngent.wire.ask import Severity
from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request
    from tests.stories.driver import Story
    from tests.stories.model import Step

    type Scripted = Callable[..., Awaitable[Story]]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the terminal under these stories is POSIX fork")


def _session_id(request: Request) -> int:
    """The session the last thing the model was read named, read the way the model reads it."""
    found = re.search(r"session_id=(\d+)", told(request))
    assert found is not None, f"nothing the model was read named a session: {told(request)!r}"
    return int(found.group(1))


def _by_session(tool: str, **arguments: object) -> Step:
    """A call of *tool* on the session the call before it opened."""
    return lambda request: calling(tool, {"session_id": _session_id(request), **arguments})


async def test_a_program_on_a_terminal_is_written_to_and_read_back(story_with: Scripted) -> None:
    """Two terminals, so the calls after the second one have to have read *that* session out of an answer."""
    shell = ["sh", "-c", "read line; echo got:$line"]
    story = await story_with(
        calling("open_pty", {"command": shell, "intention": "say it back"}),
        calling("open_pty", {"command": shell, "intention": "one more, so the two cannot be guessed apart"}),
        _by_session("write_pty", data="hello\n"),
        _by_session("read_pty", until="got:hello"),
        _by_session("close_pty"),
        text("it said hello back"),
    )

    await story.require("process")

    await story.type("say hello back")
    # Two shells and one write: a shell is a danger-level question every time (§R12 乙), and typing into
    # a live process is agreed to once.
    for _ in range(3):
        await story.answer("y")
    await story.until("turn 1 finished", "it said hello back")

    (_, first, second, written, read, closed) = story.requests()
    opened, used = _session_id(first), _session_id(second)
    assert used != opened, "two terminals were opened, so neither number is the only one that could work"
    asked = (
        f"run {shlex.join(shell)} on a terminal? "
        f"sh runs code given on its own command line, and this build does not read that code: {shell[-1]}"
    )
    assert [question.prompt for question in story.questions()] == [
        asked,
        asked,
        f"change PTY session {used}, which is still running?",
    ], "the two shells are asked about as code this build does not read, and the write once"
    assert told(written) == (f"wrote 6 chars to session {used}\nsession_id={used}\nwrote=6 chars"), (
        "the write reaches the terminal it named, in the line and in the fields"
    )
    assert f"session_id={used}" in told(read).splitlines(), "the read is of that terminal, not the other one"
    assert "got:hello" in told(read), "the read carries back what the program answered"
    assert told(closed).splitlines()[:3] == [f"closed PTY session {used}", f"session_id={used}", "alive=false"], told(
        closed
    )
    assert told(closed).splitlines()[3].startswith("exit_code="), "and what the child left behind"
    assert any(line.startswith(f"close_pty ok: closed PTY session {used}") for line in story.screen()), story.screen()


async def test_a_program_that_is_not_there_opens_a_terminal_that_says_so(story_with: Scripted) -> None:
    """A terminal whose command cannot start is a terminal, not a failed call: the complaint is output."""
    story = await story_with(
        calling("open_pty", {"command": ["plyngent-no-such-program"], "intention": "run what is not there"}),
        _by_session("read_pty", until="plyngent-pty-exec-failed"),
        _by_session("close_pty"),
        text("it never started"),
    )

    await story.require("process")

    await story.type("run what is not there")
    await story.finished()

    (_, opened, read, closed) = story.requests()
    session = _session_id(opened)
    assert f"session_id={session}" in told(read).splitlines(), "the session opened all the same"
    assert "plyngent-pty-exec-failed: " in told(read), "the terminal reports what could not be started"
    assert "alive=false" in told(read).splitlines(), "and that there is nothing left to talk to"
    assert "exit_code=127" in told(read).splitlines(), "with the code a missing program exits with"
    assert story.questions() == (), "a session that never started is not a question, on close or before"
    assert told(closed).splitlines()[:3] == [f"closed PTY session {session}", f"session_id={session}", "alive=false"]


async def test_a_terminal_that_was_never_opened_is_not_a_session(story_with: Scripted) -> None:
    story = await story_with(
        calling("read_pty", {"session_id": 99}),
        calling("close_pty", {"session_id": 99}),
        text("then there is nothing to read"),
    )

    await story.require("process")

    await story.type("read the terminal that is not there")
    await story.finished()

    (_, read, closed) = story.requests()
    assert told(read) == "unknown PTY session: 99", told(read)
    assert told(closed) == "no such PTY session: 99", told(closed)
    assert story.questions() == (), "a session that is not there changes nothing, so nothing is agreed to"


async def test_keys_are_what_a_terminal_reads_and_a_typo_is_reported(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "a terminal that keeps reading"}),
        _by_session("write_pty_keys", data="hi<enter>"),
        _by_session("write_pty", data="there\n"),
        _by_session("read_pty", until="there"),
        _by_session("write_pty_keys", data="<nope>"),
        text("done"),
    )

    await story.require("process")

    await story.type("type into it")
    await story.answer("y")
    await story.finished()

    opened, keys, written, read, bad = story.requests()[1:6]
    session = _session_id(opened)
    assert [question.prompt for question in story.questions()] == [
        f"change PTY session {session}, which is still running?"
    ], "the first write into a live process is agreed to, and the second one is not asked about again"
    assert told(keys) == (f"wrote 3 chars to session {session}\nsession_id={session}\nwrote=3 chars"), (
        "a marker is a key, not the four characters"
    )
    assert told(written) == f"wrote 6 chars to session {session}\nsession_id={session}\nwrote=6 chars"
    assert "there" in told(read), told(read)
    assert told(bad) == "unknown key <nope> (see the write_pty_keys docs for the names)", (
        "a typo is reported rather than typed into somebody's terminal"
    )


async def test_closing_a_terminal_that_is_still_running_is_a_question(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "a terminal that keeps reading"}),
        _by_session("close_pty"),
        text("left it running"),
    )

    await story.require("process")

    await story.type("close it")
    await story.answer("n")
    await story.finished()

    (_, opened, closed) = story.requests()
    session = _session_id(opened)
    assert [question.prompt for question in story.questions()] == [
        f"change PTY session {session}, which is still running?"
    ], "there is a process in there, so closing it is not the same as closing a finished one"
    assert told(closed) == f"the human did not allow changing PTY session {session}", told(closed)
    assert f"close_pty refused: the human did not allow changing PTY session {session}" in story.screen(), (
        story.screen()
    )


async def test_a_terminal_outside_the_workspace_never_starts(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "cwd": "../elsewhere", "intention": "run it next door"}),
        text("oh"),
    )

    await story.require("process")

    await story.type("open it next door")
    await story.finished()

    (_, refused) = story.requests()
    assert "escapes workspace root" in told(refused), told(refused)
    assert "../elsewhere" in told(refused), "the refusal quotes what was asked for"
    assert story.questions() == (), "a directory it may not use is not a question for the person"


async def test_a_denylisted_program_is_asked_about_before_a_terminal_starts(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["rm"], "intention": "clean something up"}),
        text("nothing started then"),
    )

    await story.require("process")

    await story.type("clean up on a terminal")
    await story.answer("n")
    await story.finished()

    (_, refused) = story.requests()
    assert [question.prompt for question in story.questions()] == [
        "run 'rm' on a terminal? it is on the command denylist"
    ], "the program decides whether anybody is asked, and it is asked before anything starts"
    assert story.questions()[0].severity is Severity.DANGER, "a program that is on a denylist is not asked lightly"
    assert told(refused) == "the human did not allow running 'rm'", told(refused)


async def test_a_conversation_s_own_room_is_a_question_asked_once(story_with: Scripted) -> None:
    """The third terminal is a question, and one yes is remembered for the conversation (§0.15, E115)."""
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "the first one"}),
        calling("open_pty", {"command": ["cat"], "intention": "the second one"}),
        calling("open_pty", {"command": ["cat"], "intention": "and one more than its room"}),
        calling("open_pty", {"command": ["cat"], "intention": "and one after that"}),
        text("four terminals, one question"),
    )

    await story.require("process")

    await story.type("open four terminals")
    await story.answer("y")
    await story.finished()

    assert [question.prompt for question in story.questions()] == [
        "open another PTY session? this one already has 2"
    ], "the third was asked about and the fourth was not: the yes is the conversation's, not the call's"
    (_, first, second, third, fourth) = story.requests()
    assert len({_session_id(first), _session_id(second), _session_id(third), _session_id(fourth)}) == 4


async def test_refusing_the_third_terminal_is_a_refusal_with_a_kind(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "the first one"}),
        calling("open_pty", {"command": ["cat"], "intention": "the second one"}),
        calling("open_pty", {"command": ["cat"], "intention": "and one more than its room"}),
        text("two is what there is room for"),
    )

    await story.require("process")

    await story.type("open three terminals")
    await story.answer("n")
    await story.finished()

    (_, _, _, too_many) = story.requests()
    assert told(too_many) == "the human did not allow another PTY session for this conversation"


async def test_a_host_with_room_for_one_terminal_says_which_number_it_hit(story_with: Scripted) -> None:
    """The host's budget is the one the program was started with, and the refusal repeats that number.

    A refusal a caller cannot act on is a refusal in the wrong place, which is why the host's budget and a
    session's are counted apart — and why the sentence carries the number: "the host is full" and "the host
    has one terminal" are one state with two different next steps (§0.11, E72).
    """
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "the one there is room for"}),
        calling("open_pty", {"command": ["cat"], "intention": "and one too many"}),
        text("one is what there is room for"),
        limits=Limits(max_sessions=1),
    )

    await story.require("process")

    await story.type("open two terminals")
    await story.finished()

    (_, first, too_many) = story.requests()
    assert _session_id(first) > 0, "the first one opened"
    assert told(too_many) == "PTY session limit reached (1); close an idle session first", told(too_many)


async def test_the_humans_answer_reaches_the_terminal_and_is_reported_nowhere_else(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "a terminal to put a secret into"}),
        _by_session("ask_into_pty", message="the passphrase", secret=True, intention="it must not be in the history"),
        text("wrote it"),
    )

    await story.require("process")

    await story.type("ask me for the passphrase")
    await story.answer("s3cret")
    await story.finished()

    (_, opened, asked) = story.requests()
    session = _session_id(opened)
    assert told(asked).splitlines() == [
        f"wrote the human's answer to session {session}",
        f"session_id={session}",
        "wrote=true",
        "secret=true",
        "submit=true",
        "source=human",
    ], "what the kernel did with the answer, and not the answer"
    assert not any("s3cret" in line for line in story.screen() + story.notices()), (
        "the secret went from the person to the terminal: it is not on the screen they are reading"
    )


async def test_an_answer_nobody_gave_writes_nothing(story_with: Scripted) -> None:
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "a terminal to put a secret into"}),
        _by_session("ask_into_pty", message="the passphrase", secret=True, intention="it must not be in the history"),
        text("fine"),
    )

    await story.require("process")

    await story.type("ask me for the passphrase")
    await story.decline()
    await story.finished()

    (_, _, asked) = story.requests()
    assert told(asked) == "the human gave no answer, so nothing was written", told(asked)
