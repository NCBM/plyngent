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
from tests.stories.driver import SETTLE_PAUSE, SETTLE_TRIES

if TYPE_CHECKING:
    from pathlib import Path
    from typing import TextIO


async def test_a_pipe_is_a_session_from_a_command_to_the_end_of_its_input(tmp_path: Path) -> None:
    """One command, one message, and the end of the input, over a descriptor rather than a keyboard.

    What the person reads is what a terminal would have read, less the drawing: the command's listing
    goes to the notices because that is where a complaint goes, and the refusal goes to the screen
    because that is where what the kernel *said* goes. The question is asked progressively — the pipe
    is fed one line at a time and the story waits for the answer to land — because a file written all
    at once would end the program before it had drawn anything.
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
            await _until(out, "turn refused: this kernel has no model installed")

            feed.close()
            code = await running
        finally:
            await kernel.close()

    assert code == 0, "the end of the input is a goodbye, not a failure"
    assert "commands you can type:" in err.getvalue()
    assert out.getvalue().startswith("turn refused:"), "and the message it sent reached the kernel"


async def _until(stream: TextIO, wanted: str) -> None:
    """Wait until this line has been written to that stream.

    The same bounded wait the story driver uses, and for the same reason: a verb that reads on a thread
    makes no progress while the loop spins without yielding time, so waiting is a number of short
    pauses rather than a number of turns.
    """
    for _ in range(SETTLE_TRIES):
        if wanted in stream.getvalue():
            return
        await asyncio.sleep(SETTLE_PAUSE)
    msg = f"{wanted!r} never arrived; the stream said {stream.getvalue()!r}"
    raise AssertionError(msg)
