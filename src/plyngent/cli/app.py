"""The loop: three tasks, one cursor, and one way out.

A client of this kernel has exactly three jobs, and they run as three tasks because each one waits on
something different:

* **events** — read what the kernel says, draw it, and tell the transport how far it got. The ack is
  one number per message, and for a folded batch that number covers every item in it, because the
  batch is numbered by its last one.
* **questions** — read what the kernel is waiting on, take the cursor for it, and send back exactly
  one answer. This is why the editor is the same object the line loop uses: a question may take the
  cursor from a line mid-typing (R2.4 C3).
* **lines** — read one line, decide what it meant, and send it. Plain text is a message; a slash word
  is a command (:mod:`plyngent.cli.grammar`).

Leaving has three entrances and one exit. The connection refusing a command, the event lane ending
(the kernel left), and Ctrl+D all end the loop; the first says the connection is gone (exit code 1)
and the other two are a normal goodbye (0). Before closing, the client says *why* it is leaving with a
``Cancel`` — the kernel cannot tell "left" from "slow", so it has to be told (R2.1 D15).

What this file deliberately does not do: draw a prompt, decide how a line looks, or know what a
command's arguments are. Those are the editor's and the grammar's, and this loop only routes.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from plyngent.cli.grammar import Catalog, Composes, Help, Runs, Wrong, decode
from plyngent.cli.line import Cancelled, EndOfInput, Preempted, Typed
from plyngent.cli.render import render
from plyngent.wire.command import Cancel, Open, Submit
from plyngent.wire.flow import Ack
from plyngent.wire.scope import CancelReason

if TYPE_CHECKING:
    from typing import TextIO

    from plyngent.cli.line import Line, LineEditor
    from plyngent.wire.link import KernelLink

#: What a normal goodbye returns.
LEFT = 0

#: What a goodbye through a connection that is already gone returns.
GONE = 1


class App:
    """One client of one kernel: its three tasks, and the exit code they add up to."""

    client: KernelLink
    session: str
    editor: LineEditor
    catalog: Catalog
    _out: TextIO
    _err: TextIO
    _leaving: asyncio.Event

    def __init__(
        self,
        client: KernelLink,
        *,
        session: str,
        catalog: Catalog,
        editor: LineEditor,
        out: TextIO,
        err: TextIO,
    ) -> None:
        self.client = client
        self.session = session
        self.editor = editor
        self.catalog = catalog
        self._out = out
        self._err = err
        self._leaving = asyncio.Event()

    async def run(self) -> int:
        """Serve the connection until somebody says goodbye, then say why before closing it.

        Opening the session is the first thing a client does, and it is a command rather than an
        argument because a session is admitted into the kernel's assembly, not handed to a process
        (R2.3 第 8 步).
        """
        if not self.client.submit(Open(session=self.session)):
            return GONE
        background = [asyncio.create_task(self._events()), asyncio.create_task(self._questions())]
        try:
            return await self._lines()
        finally:
            for task in background:
                _ = task.cancel()
            _ = await asyncio.gather(*background, return_exceptions=True)
            _ = self.client.submit(Cancel(session=self.session, scope=None, reason=CancelReason.CLIENT_GONE))
            self.client.close()

    # -- what the kernel says ------------------------------------------------------

    async def _events(self) -> None:
        """Draw every event, then tell the transport how far this client has got.

        The lane ending is not a failure: ``KernelLeaving`` is a message like any other, and the end of
        the stream after it is the kernel's way of saying the connection is over (R2.1 D20).
        """
        async for event in self.client.events():
            self._say(self._out, *render(event, focus=self.session))
            self.client.ack(Ack(seq=event.seq))
        self._leaving.set()

    async def _questions(self) -> None:
        """Answer what the kernel asks, one at a time, through the one cursor."""
        async for question in self.client.questions():
            _ = self.client.answer(await self.editor.ask(question))

    # -- what the human says -------------------------------------------------------

    async def _lines(self) -> int:
        """Read, decide, send — until the human is done or the connection is."""
        while True:
            line = await self._next_line()
            if line is None:
                return LEFT
            match line:
                case Typed(text=text):
                    if text.strip():
                        code = await self._said(text)
                        if code is not None:
                            return code
                case EndOfInput():
                    return LEFT
                case Cancelled():
                    _ = self.client.submit(Cancel(session=self.session, scope=None, reason=CancelReason.USER))
                case Preempted():
                    # A question took the cursor; it has already asked, and the editor holds the buffer.
                    continue

    async def _said(self, line: str) -> int | None:
        """One line's meaning, and the exit code it earns if the connection was gone."""
        match decode(line, self.catalog, session=self.session):
            case Runs(command=command):
                return None if self.client.submit(command) else GONE
            case Composes(session=session):
                composed = await self.editor.compose()
                if isinstance(composed, Typed):
                    return None if self.client.submit(Submit(session=session, text=composed.text)) else GONE
                return None
            case Help(text=text):
                self._say(self._err, text)
                return None
            case Wrong(text=text):
                self._say(self._err, text)
                return None

    async def _next_line(self) -> Line | None:
        """One line, or ``None`` when the kernel left while the human was still typing."""
        reading = asyncio.create_task(self.editor.read_line())
        left = asyncio.create_task(self._leaving.wait())
        done, _ = await asyncio.wait({reading, left}, return_when=asyncio.FIRST_COMPLETED)
        if reading in done:
            _ = left.cancel()
            _ = await asyncio.gather(left, return_exceptions=True)
            return reading.result()
        _ = reading.cancel()
        _ = await asyncio.gather(reading, return_exceptions=True)
        return None

    def _say(self, stream: TextIO, *lines: str) -> None:
        """Write whole lines and flush: an event is worth showing before the next one arrives."""
        for line in lines:
            _ = stream.write(f"{line}\n")
        _ = stream.flush()
