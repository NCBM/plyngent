"""The plain lane in the real loop: a pipe is a whole session, from a command to a goodbye.

The stories around this one put a person in front of the loop and let them type; this one puts a file
there instead. Nothing else changes — the same ``App``, the same kernel, the same real connection — and
the claim is that the loop cannot tell: one editor in place of another, and the program still opens a
session, still says what the kernel said, and still returns 0 when its input ends.

The stream is a real pipe on a real thread, because that is the case the lane exists for (a script, a
CI job, a supervisor with no terminal), and a ``StringIO`` would not prove that ``readline`` off a file
descriptor reaches the loop.
"""

from __future__ import annotations

import asyncio
import io
import os
from typing import TYPE_CHECKING

from plyngent.boot import install
from plyngent.cli.app import App
from plyngent.cli.grammar import build
from plyngent.cli.line import PlainEditor
from plyngent_creative.model import Held, ScriptedModel, text
from tests.stories.driver import SETTLE_PAUSE, SETTLE_TRIES

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO


async def test_a_pipe_is_a_session_from_a_command_to_the_end_of_its_input(tmp_path: Path) -> None:
    """One command, one message, and the end of the input, over a descriptor, not a keyboard.

    What the person reads is what a terminal would have read, less the drawing: the command's listing
    goes to the notices because that is where a complaint goes, and the refusal goes to the screen
    because that is where what the kernel *said* goes. The pipe is fed one line at a time and the story
    waits for each answer to land, which keeps the two claims apart: what a command answers and what a
    message answers. A pipe written in one go is the story below.
    """
    reading, writing = os.pipe()
    with os.fdopen(reading, "r", encoding="utf-8") as stream, os.fdopen(writing, "w", encoding="utf-8") as feed:
        kernel = await install(tmp_path / "workspace", journal=tmp_path / "journal.db")
        try:
            out, err = io.StringIO(), io.StringIO()
            editor = PlainEditor(session="main", stream=stream, err=err)
            catalog = build(kernel.client.manifest, session="main")
            app = App(kernel.client, session="main", catalog=catalog, editor=editor, out=out, err=err)
            running = asyncio.create_task(app.run())

            _ = feed.write("/help\n")
            _ = feed.flush()
            await _until(err, "anything else you type is a message")

            _ = feed.write("look at the README\n")
            _ = feed.flush()
            await _until(out, "turn refused: this conversation has no model to ask")

            feed.close()
            code = await running
        finally:
            await kernel.close()

    assert code == 0, "the end of the input is a goodbye, not a failure"
    assert "commands you can type:" in err.getvalue()
    assert out.getvalue().startswith("turn refused:"), "and the message it sent reached the kernel"


async def test_a_pipe_written_in_one_go_still_reads_what_it_asked_for(tmp_path: Path) -> None:
    """The whole input at once, then the end of it: the answer the last line asked for is still read.

    This is the case the loop has to wait for, and the reason it is a case at all: a command has no
    receipt and its answer is an event, so a program that left the moment the last line was read would
    drop the answer that line asked for. Here the pipe is written and closed before the program has said
    anything, so everything the person said is waiting when the loop starts.

    The listing arrives once: what the line asked for is drawn, and the last listing — the one the
    goodbye waits on — is this client's own business and is dropped.
    """
    reading, writing = os.pipe()
    with os.fdopen(reading, "r", encoding="utf-8") as stream, os.fdopen(writing, "w", encoding="utf-8") as feed:
        _ = feed.write("/sessions\n")
        _ = feed.flush()
        feed.close()
        kernel = await install(tmp_path / "workspace", journal=tmp_path / "journal.db")
        try:
            out, err = io.StringIO(), io.StringIO()
            editor = PlainEditor(session="main", stream=stream, err=err)
            catalog = build(kernel.client.manifest, session="main")
            app = App(kernel.client, session="main", catalog=catalog, editor=editor, out=out, err=err)
            code = await app.run()
        finally:
            await kernel.close()

    assert code == 0
    asked = out.getvalue()
    assert asked.count("sessions:") == 1, "the answer was drawn once, and the goodbye's was not"
    assert "main" in asked, "and it is the listing the person asked for"


async def test_a_pipe_waits_for_the_turn_a_message_started(tmp_path: Path) -> None:
    """The end of the input is not the end of the work, because a message is answered by a turn.

    A command has a receipt the goodbye can wait on — the listing — and a turn has none: it is a task the
    kernel started and reports the end of by an event, so a program that left when the last line was read
    would drop the answer that line asked for. The model here is held until this story lets it go, which
    is what makes the difference observable: the loop is still there, waiting, and what it is waiting for
    is the end of the turn, not another line.
    """
    reading, writing = os.pipe()
    with os.fdopen(reading, "r", encoding="utf-8") as stream, os.fdopen(writing, "w", encoding="utf-8") as feed:
        held = Held(text("about time"))
        kernel = await install(tmp_path / "workspace", journal=tmp_path / "journal.db", model=ScriptedModel([held]))
        try:
            out, err = io.StringIO(), io.StringIO()
            editor = PlainEditor(session="main", stream=stream, err=err)
            catalog = build(kernel.client.manifest, session="main")
            app = App(kernel.client, session="main", catalog=catalog, editor=editor, out=out, err=err)
            running = asyncio.create_task(app.run())

            _ = feed.write("a thought\n")
            _ = feed.flush()
            feed.close()
            await _until(out, "turn 1 started (chat)")
            await asyncio.sleep(SETTLE_PAUSE)
            assert not running.done(), "the end of the input is not the end of the turn it started"

            held.ready.set()
            code = await asyncio.wait_for(running, timeout=SETTLE_TRIES * SETTLE_PAUSE)
        finally:
            await kernel.close()

    assert code == 0
    assert "about time" in out.getvalue(), "the answer the message asked for was read before the program left"


async def _until(stream: TextIO, wanted: str) -> None:
    """Wait until this line has been written to that stream.

    The same bounded wait the story driver uses, and for the same reason: a verb that reads on a thread
    makes no progress while the loop spins without yielding time, so waiting is a number of short
    pauses, not a number of turns.
    """
    for _ in range(SETTLE_TRIES):
        if wanted in stream.getvalue():
            return
        await asyncio.sleep(SETTLE_PAUSE)
    msg = f"{wanted!r} never arrived; the stream said {stream.getvalue()!r}"
    raise AssertionError(msg)
