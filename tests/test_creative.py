"""The plugin distribution: the run it drives a kernel with, and what that run has to say.

This is the half of the distribution that a plugin author imports and a check drives, so what it is tested
for is what a client is promised: the conversation it opens, the turn it runs, and the events the answer
arrives in. Nothing here reads a rendering — the words a person would see are the frontend's business, and
a plugin author asserts on the contract (E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.failure import Failure
from plyngent.wire.command import Compact
from plyngent.wire.event import (
    AssistantDelta,
    AssistantText,
    Pruned,
    Refusal,
    ToolCalled,
    ToolReturned,
    TurnFinished,
    TurnKind,
    TurnRefused,
    TurnStarted,
)
from plyngent_creative.model import Cut, Failing, ScriptedModel, text
from plyngent_creative.session import Run
from plyngent_creative.steps import ScriptError, arrived, parse

if TYPE_CHECKING:
    from pathlib import Path

#: How long one test waits for the turn it started. The kernel answers a scripted model at once, and a
#: run that never finished is a test that must fail at a deadline and not hang the suite.
WAIT = 10.0


async def test_a_run_drives_one_turn_and_keeps_what_it_said(tmp_path: Path) -> None:
    """A conversation opened through the contract, a message, and the answer, as the kernel said them.

    The three events are the whole sentence: the turn began, the model's words arrived, and the turn ended
    on its own. They are what an expectation a plugin author writes is matched against, so the run that
    carries them is the one thing that has to work before any of the tools that drive it mean anything.
    """
    run = await Run.open(
        tmp_path / "workspace",
        model=ScriptedModel([text("hello")]),
        journal=tmp_path / "journal.db",
    )
    try:
        run.say("hi")
        await run.finished(patience=WAIT)
        said = run.events
    finally:
        await run.close()

    assert [one.turn for one in said if isinstance(one, TurnStarted)] == [1]
    assert [one.text for one in said if isinstance(one, AssistantText)] == ["hello"]
    assert [one.reason for one in said if isinstance(one, TurnFinished)] == [""]


async def test_a_turn_the_kernel_refused_is_not_a_turn_that_finished(tmp_path: Path) -> None:
    """A kernel with no model says why it will not run a turn, and waiting for that turn's end repeats it.

    The refusal has no turn number because no turn began, so it is the one event that answers "how did
    that go" without a turn to point at — and a check that treated it as a turn that ended would pass on
    a conversation where nothing ran at all.
    """
    run = await Run.open(tmp_path / "workspace", journal=tmp_path / "journal.db")
    try:
        run.say("hi")
        with pytest.raises(RuntimeError, match=str(Refusal.NO_MODEL)):
            await run.finished(patience=WAIT)
        said = run.events
    finally:
        await run.close()

    assert [one.reason for one in said if isinstance(one, TurnRefused)] == [Refusal.NO_MODEL]


async def test_a_reply_that_breaks_part_way_is_the_pieces_and_then_the_failure(tmp_path: Path) -> None:
    """An endpoint that times out mid-answer says part of a sentence and then stops, and both are seen.

    What a listener read is the fragments that arrived; what the turn stands behind is the failure, so the
    turn ends with it and nothing claims the answer was whole (E145, E184).
    """
    run = await Run.open(
        tmp_path / "workspace",
        model=ScriptedModel(
            [Cut(pieces=("half ", "an answer"), failure=Failure.TIMEOUT, says="the endpoint stopped")],
        ),
        journal=tmp_path / "journal.db",
    )
    try:
        run.say("hi")
        with pytest.raises(RuntimeError, match="did not finish"):
            await run.finished(patience=WAIT)
        said = run.events
    finally:
        await run.close()

    assert [one.text for one in said if isinstance(one, AssistantDelta)] == ["half ", "an answer"]
    assert [one.failure for one in said if isinstance(one, TurnFinished)] == [str(Failure.TIMEOUT)]


# -- what a script says, which needs no kernel to read ----------------------------------------


def test_a_step_holds_the_three_things_a_script_says() -> None:
    """One step is what the model answers, what is sent, and what has to arrive, and a step may say all three.

    They are three fields of one value and not three kinds of step, because the order they happen in is
    fixed — the model answers, the line is sent, what arrives is looked for — and a value that carried its
    own order would be a second way to say the same thing.
    """
    (step,) = parse(
        [
            {
                "model": {"text": "asking", "calls": [{"tool": "read_file", "arguments": {"path": "a.txt"}}]},
                "send": "go on",
                "expect": {"tag": "tool-called", "fields": {"tool": "read_file"}},
            }
        ]
    )

    assert step.model is not None
    assert (step.model.text, [call.tool for call in step.model.calls]) == ("asking", ["read_file"])
    assert step.send == "go on"
    assert step.expect is not None and step.expect.tag == "tool-called"


def test_a_command_is_the_shape_the_contract_already_has() -> None:
    """A line starting with a slash is decoded as the command it names, so the contract keeps one grammar.

    ``{"cmd": "compact", "text": …}`` is a command object and not a second spelling of one: a script writes
    the same value a frontend sends, and a word this build does not have is refused where the script is read.
    """
    (step,) = parse([{"send": {"cmd": "compact", "text": "so far"}}])

    assert isinstance(step.send, Compact) and step.send.text == "so far"

    with pytest.raises(ScriptError, match="the contract does not have"):
        parse([{"send": {"cmd": "no-such-command"}}])


def test_a_word_a_step_does_not_have_is_refused_by_name() -> None:
    """A key nothing reads is a step that says less than it looks like it says, so it is refused."""
    with pytest.raises(ScriptError, match="says sent, and the words here are"):
        parse([{"sent": "go on"}])
    with pytest.raises(ScriptError, match="is empty"):
        parse([{}])
    with pytest.raises(ScriptError, match="is a str where an object belongs"):
        parse(["go on"])


def test_a_failure_is_one_of_the_words_the_contract_knows() -> None:
    """The kinds of trouble are the kernel's closed set, so a script cannot invent a thirteenth."""
    (step,) = parse([{"model": {"fail": "timeout", "says": "the endpoint stopped"}}])

    assert step.model is not None
    assert isinstance(step.model.scripted(), Failing)

    with pytest.raises(ScriptError, match="the kinds of trouble are"):
        parse([{"model": {"fail": "sad"}}])


def test_an_answer_that_breaks_says_how_much_arrived() -> None:
    """``cut_after`` and ``failure`` are one fact said twice, and a script that says one of them is refused.

    A stream that broke is the pieces somebody read and then the failure, so neither half alone describes
    what the model did: the check is that both are there, and the count is at least one piece.
    """
    (step,) = parse([{"model": {"text": "half an answer", "cut_after": 2, "failure": "timeout"}}])

    assert step.model is not None
    scripted = step.model.scripted()
    assert isinstance(scripted, Cut) and scripted.pieces == ("half ", "an ")

    with pytest.raises(ScriptError, match="no failure"):
        parse([{"model": {"text": "x", "cut_after": 1}}])
    with pytest.raises(ScriptError, match="no cut_after"):
        parse([{"model": {"failure": "timeout"}}])
    with pytest.raises(ScriptError, match="at least one piece"):
        parse([{"model": {"text": "x", "cut_after": 0, "failure": "timeout"}}])


def test_an_expectation_is_a_tag_and_a_subset_of_fields() -> None:
    """The fields are a subset all the way down, which is what lets an author name one field of an event.

    The tags and the words inside are the ones the contract spells, so an author writes ``"kind": "chat"``
    and not the kernel's own enum: what is compared is the value the wire carries.
    """
    (step,) = parse([{"expect": {"tag": "tool-returned", "fields": {"tool": "read_file", "status": "ok"}}}])

    assert step.expect is not None
    assert step.expect.wants(ToolReturned(tool="read_file", status="ok", summary="done"))
    assert not step.expect.wants(ToolReturned(tool="read_file", status="failed", summary="done"))
    assert not step.expect.wants(TurnFinished())
    assert "tool-returned" in step.expect.says()

    (kind,) = parse([{"expect": {"tag": "turn-started", "fields": {"kind": "chat"}}}])
    assert kind.expect is not None and kind.expect.wants(TurnStarted(turn=1, kind=TurnKind.CHAT))


def test_a_list_in_an_expectation_is_equal_and_not_a_subset() -> None:
    """A list says the whole of what it means, because a prefix of a list is a different list."""
    (step,) = parse([{"expect": {"tag": "pruned", "fields": {"targets": ["a"]}}}])

    assert step.expect is not None
    assert step.expect.wants(Pruned(targets=("a",)))
    assert not step.expect.wants(Pruned(targets=("a", "b")))


def test_what_arrived_is_one_line_per_event_with_the_fields_a_line_can_hold() -> None:
    """A step that matched nothing shows what did arrive, and a structure on a line would say nothing."""
    shown = arrived(
        [
            TurnStarted(turn=1, kind=TurnKind.CHAT),
            ToolCalled(turn=1, call="read_file", tool="read_file"),
        ]
    )

    first, second, *_ = shown.splitlines()
    assert first.startswith("turn-started ") and "turn=1" in first and "kind='chat'" in first
    assert second.startswith("tool-called ") and "tool='read_file'" in second
    assert "items" not in shown
