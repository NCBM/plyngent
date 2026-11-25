"""The plugin distribution: the run it drives a kernel with, and what that run has to say.

This is the half of the distribution that a plugin author imports and a check drives, so what it is tested
for is what a client is promised: the conversation it opens, the turn it runs, and the events the answer
arrives in. Nothing here reads a rendering — the words a person would see are the frontend's business, and
a plugin author asserts on the contract (E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from plyngent_creative.model import ScriptedModel, text
from plyngent_creative.session import Run

from plyngent.wire.event import AssistantText, Refusal, TurnFinished, TurnRefused, TurnStarted

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
