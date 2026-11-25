"""What a process that died in the middle of a turn leaves, and what the next one does about it.

The state these stories read cannot be reached by letting a turn run: it is a chain whose last answer
asked for calls that nothing answered, which is a process killed between the answer and the results. So
the rows are written here, exactly as the turn writes them (:meth:`~plyngent.kernel.turn.TurnRunner.dispatch`
appends the answer, then a dispatch row per call, then the results), and everything after that is the real
program reading a real journal (E192).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.journal import SqliteTrail
from plyngent.kernel.model import Assistant, Call, Reply, Result, User
from plyngent.kernel.trail import Dispatched, TurnEnded, kind_of
from plyngent.kernel.turn import INTERRUPTED, UNENDED, UNSTARTED, mend
from plyngent_creative.model import ScriptedModel, text
from tests.stories.driver import Story

if TYPE_CHECKING:
    from plyngent.kernel.model import Request

#: The calls the killed turn's answer asked for, in the order it asked. The first came back before the
#: process died, the second had been handed to a tool, and the third never was.
ANSWERED, HANDED, NEVER = "answered", "handed", "never"


def _call(one: str) -> Call:
    return Call(id=one, tool="read_file", arguments=json.dumps({"path": "notes.txt"}))


async def _crashed(journal: Path) -> None:
    """Write the chain a process killed in the middle of a batch leaves behind, at turn 1, unfinished."""
    trail = SqliteTrail(journal)
    await trail.open()
    try:
        # The turn's number is the log's to allocate, and a killed process had already taken it — so the
        # counter is moved here the way `begin_turn` moves it, and the next turn is not this one.
        turn = await trail.next_turn("main")
        said = await trail.append("main", User(text="read the note"), turn=turn)
        asked = await trail.append(
            "main",
            Assistant(text="let me read it", calls=(_call(ANSWERED), _call(HANDED), _call(NEVER))),
            begin_after=said.id,
            turn=turn,
        )
        shown = await trail.append(
            "main", Dispatched(turn=turn, call=ANSWERED, kind="chat"), begin_after=asked.id, turn=turn
        )
        done = await trail.append(
            "main", Result(call=ANSWERED, tool="read_file", text="hi"), begin_after=shown.id, turn=turn
        )
        _ = await trail.append("main", Dispatched(turn=turn, call=HANDED, kind="chat"), begin_after=done.id, turn=turn)
    finally:
        await trail.close()


def _refuses(request: Request) -> Reply:
    """A provider that reads a call nobody answered as a broken conversation, the way most of them do."""
    answered = {message.call for message in request.messages if isinstance(message, Result)}
    for message in request.messages:
        if isinstance(message, Assistant):
            dangling = [call.id for call in message.calls if call.id not in answered]
            assert not dangling, f"a provider would refuse this conversation: {dangling} was never answered"
    return text("carried on")


async def test_a_turn_a_killed_process_left_is_repaired_before_the_next_request(tmp_path: Path, transport: str) -> None:
    """The first request after the crash carries answers to every call, so a provider can read it.

    The repair is not a second driver: the turn is closed with the two sentences the log proves, and the
    conversation then goes on from the repaired end.
    """
    journal = tmp_path / "journal.db"
    await _crashed(journal)

    story = await Story.start(
        tmp_path / "workspace", transport=transport, model=ScriptedModel([_refuses]), journal=journal
    )
    try:
        await story.type("carry on")
        await story.finished(2)

        rows = await story.host.trail.entries(story.session)
        results = {row.payload.call: row.payload for row in rows if isinstance(row.payload, Result)}
        ends = [row.payload for row in rows if isinstance(row.payload, TurnEnded)]

        assert results[ANSWERED].text == "hi", "the call that came back is left as it was"
        assert results[HANDED].text == INTERRUPTED, "a call the process handed over may have run"
        assert results[NEVER].text == UNSTARTED, "a call it never handed over cannot have"
        assert [(one.turn, one.reason, one.kind) for one in ends] == [(1, UNENDED, "chat"), (2, "", "chat")], (
            "the interrupted turn now says it ended, and says which kind of turn it was"
        )
    finally:
        await story.close()


async def test_a_conversation_left_mid_turn_is_repaired_once(tmp_path: Path) -> None:
    """A function of the log: the second answer is nothing, and nothing about the log moved."""
    journal = tmp_path / "journal.db"
    await _crashed(journal)

    trail = SqliteTrail(journal)
    await trail.open()
    try:
        first = await mend(trail, "main")
        rows = await trail.entries("main")
        second = await mend(trail, "main")

        assert [kind_of(row.payload) for row in first] == ["tool-result", "tool-result", "turn-ended"]
        assert second == (), "a turn that says it ended is a turn nothing is owed"
        assert await trail.entries("main") == rows, "and repairing twice wrote nothing the second time"
    finally:
        await trail.close()


async def test_the_repair_leaves_a_conversation_that_ended_alone(tmp_path: Path) -> None:
    """Every turn with an end row is skipped, which is what makes opening a conversation cheap."""
    journal = tmp_path / "journal.db"
    trail = SqliteTrail(journal)
    await trail.open()
    try:
        said = await trail.append("main", User(text="hello"), turn=1)
        answered = await trail.append("main", Assistant(text="hi"), begin_after=said.id, turn=1)
        _ = await trail.append("main", TurnEnded(turn=1, kind="chat"), begin_after=answered.id, turn=1)
        rows = await trail.entries("main")

        assert await mend(trail, "main") == ()
        assert await trail.entries("main") == rows
    finally:
        await trail.close()
