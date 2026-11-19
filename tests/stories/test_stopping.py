"""What a program says at the moment it takes something with it (E151).

Two writers and one rule between them: the sentence is written **when the loss is true** — a process that is
going down, or a broker that has just been replaced — rather than guessed at by the next start, and it is
written only into a conversation that really loses something. What a story can show is the row, read back by
the next process, and the fact that a conversation which lost nothing was told nothing.

A process that is killed says nothing at all, and that is not a gap in this file: it is the accepted cost of
saying it here, because "the last one died" is only provable by a timestamp this tree does not keep.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from composat import Declaration

from plyngent.kernel.model import Note
from plyngent.kernel.pty.broker import PtyBroker
from plyngent.kernel.trail import Notice
from tests.stories.driver import Story
from tests.stories.model import Held, ScriptedModel, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    type Scripted = Callable[..., Awaitable[Story]]


async def _said(story: Story) -> tuple[Notice, ...]:
    """The rows this conversation's chain holds that a model is told something in."""
    rows = await story.host.trail.entries(story.session)
    return tuple(entry.payload for entry in rows if isinstance(entry.payload, Notice))


async def test_a_conversation_is_told_the_scratch_it_loses_with_the_process(tmp_path: Path, transport: str) -> None:
    """A scratch directory is out of reach once the kernel stops, so the row says so — and only that (E151).

    The other half of the rule is here as well: a conversation that never opened a terminal is not told
    about terminals, because a sentence a model cannot act on is noise where it reads most carefully.
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("new_temporary_workspace", {"prefix": "story"}), text("made one")]),
        journal=journal,
    )
    try:
        await first.require("fs")
        await first.type("make me somewhere to work")
        await first.answer("y")
        await first.finished()

        assert await first.stop() == 0, "the program left the way a person leaves it"
    finally:
        await first.close()

    second = await Story.start(root, transport=transport, model=ScriptedModel([]), journal=journal)
    try:
        (said,) = await _said(second)

        assert said.text.startswith("[notice from harness]"), "the model reads a reminder, not a tool result"
        assert "temporary workspace" in said.text, said.text
        assert "out of reach" in said.text, "which is the certain half of it"
        assert "may have been removed" in said.text, "and the uncertain half is said as uncertain"
        assert "terminal" not in said.text, "a conversation that had none is not told that it lost one"
        assert said.summary == "1 temporary workspace out of reach", "and a person reads one line about it"
    finally:
        await second.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the terminal under this story is POSIX fork")
async def test_a_conversation_is_told_the_terminals_that_end_with_the_process(tmp_path: Path, transport: str) -> None:
    """A child on a terminal dies with the process that holds it, and the row counts them before it does."""
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("open_pty", {"command": ["cat"], "intention": "a shell"}), text("it is open")]),
        journal=journal,
    )
    try:
        await first.require("process")
        await first.type("open a shell")
        await first.finished()

        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(root, transport=transport, model=ScriptedModel([]), journal=journal)
    try:
        (said,) = await _said(second)

        assert "the 1 terminal session you have open end with it" in said.text, said.text
        assert "temporary workspace" not in said.text, "and nothing about scratch it never asked for"
        assert said.summary == "1 terminal session gone with the process"
    finally:
        await second.close()


@pytest.mark.skipif(sys.platform == "win32", reason="the terminal under this story is POSIX fork")
async def test_a_replaced_terminal_broker_is_said_before_a_model_calls_an_old_id(story_with: Scripted) -> None:
    """The refusal for a dead id is still there; what this ends is the call a model would spend finding it.

    The broker is replaced while the second request is in flight, so the sentence is written at the next
    boundary — the third round's reminder — rather than in the middle of one. What the request carries is
    the one thing that changed: a note, and no call of ours in between.
    """
    held = Held(calling("read", {"path": "notes.txt"}))
    story = await story_with(
        calling("open_pty", {"command": ["cat"], "intention": "a shell"}),
        held,
        text("understood"),
    )
    await story.require("process")

    await story.type("open a shell")
    await story.settle()

    _ = await story.connection.assembly.replace(Declaration(PtyBroker).key, Declaration(PtyBroker))
    held.ready.set()
    await story.finished()

    _first, second, third = story.requests()
    assert not [message for message in second.messages if isinstance(message, Note)], (
        "nothing was said while the broker was still the one that held the terminal"
    )
    notes = [message for message in third.messages if isinstance(message, Note)]
    assert len(notes) == 1, f"one sentence, at the first boundary after it: {third.messages!r}"
    assert "the 1 terminal session you had open are gone" in notes[0].text, notes[0].text
    assert "Every session id you were given for one is dead" in notes[0].text
    assert any("1 terminal session lost when the terminal broker was replaced" in line for line in story.screen()), (
        f"and the person reads one line about it: {story.screen()!r}"
    )
