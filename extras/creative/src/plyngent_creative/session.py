"""One kernel this distribution brings up and drives, and every event it said.

A plugin author needs what a story has: a real kernel, a real connection, and a model that answers what
the script says. This module is that without a terminal — it brings up the composition root on a root of
its own, opens one conversation, sends what it is told to send, and keeps what came back. Expectations are
read from those events, so what an author checks is the contract their plugin speaks and not a rendering
of it (E184).

It names :mod:`plyngent.boot`, :mod:`plyngent.wire` and :mod:`plyngent.kernel.model`, because those are
the three doors a client has: the composition root hands out a link, the link speaks the contract, and a
model is what a caller scripts. ``tests/test_structure.py`` pins that reach, since a fourth name here
would be a kernel change that breaks somebody who installed this distribution.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.boot import install
from plyngent.wire.command import Open, Submit
from plyngent.wire.event import TurnFinished, TurnRefused

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from composat import Declaration

    from plyngent.boot import Installed
    from plyngent.kernel.model import Model
    from plyngent.wire.command import AnyCommand
    from plyngent.wire.event import AnyEvent
#: The conversation a run opens when nobody says otherwise. It is a word a person could type and not a
#: number, because the id is what a command names and nothing about this one is special (E184).
SESSION = "debug"


class GoneError(RuntimeError):
    """The kernel had already stopped, so what was sent went nowhere."""


class Run:
    """A kernel in one root, one conversation in it, and every event it has said.

    The events are read for as long as the connection lives, on a task of their own, because the kernel
    says them when it says them: a client that read them inside an expectation could not wait for one that
    has not arrived yet, and one that read them after the fact would miss what a question did.
    """

    root: Path
    session: str
    #: What is told about every event as it arrives, and ``None`` is a run whose events are only kept. A
    #: driver that is stepping through a turn says what happened while the turn is still running, which is
    #: the difference between reading a transcript and watching one (E184).
    on_event: Callable[[AnyEvent], None] | None
    _kernel: Installed
    _events: list[AnyEvent]
    _arrived: asyncio.Event
    _reading: asyncio.Task[None]
    _cursor: int

    def __init__(
        self,
        kernel: Installed,
        root: Path,
        session: str,
        *,
        on_event: Callable[[AnyEvent], None] | None = None,
    ) -> None:
        self.root = root
        self.session = session
        self.on_event = on_event
        self._kernel = kernel
        self._events = []
        self._arrived = asyncio.Event()
        self._cursor = 0
        self._reading = asyncio.create_task(self._read())

    @classmethod
    async def open(
        cls,
        root: Path | str,
        *,
        model: Model | None = None,
        journal: Path | str | None = None,
        session: str = SESSION,
        components: Iterable[Declaration] = (),
        on_event: Callable[[AnyEvent], None] | None = None,
    ) -> Run:
        """Bring up a kernel in ``root`` and open one conversation in it.

        ``model`` is installed the way any launcher installs one: ``None`` is a kernel that refuses turns,
        which is its own expectation. ``components`` is what a plugin brings — the declarations a
        distribution answers — so a run can be about the plugin under test and not about this program.
        """
        path = Path(root)
        _ = await asyncio.to_thread(path.mkdir, parents=True, exist_ok=True)
        started = await install(path, model=model, journal=journal, components=components)
        run = cls(started, path, session, on_event=on_event)
        run.submit(Open(session=session))
        return run

    # -- what a script sends -------------------------------------------------------

    def submit(self, command: AnyCommand) -> None:
        """Hand the kernel one command, the way a client does."""
        if not self._kernel.client.submit(command):
            msg = "the kernel had already stopped, so this command went nowhere"
            raise GoneError(msg)

    def say(self, text: str) -> None:
        """Say one thing in this conversation, as a person typing a line and pressing Enter."""
        self.submit(Submit(session=self.session, text=text))

    def send(self, what: str | AnyCommand) -> None:
        """Send what a step says: a line somebody types, or a command the contract already spells.

        The two are one door because they are one thing to a script — "do this now" — and a line is the
        shape a person's words have, which is why it needs no object around it.
        """
        if isinstance(what, str):
            self.say(what)
        else:
            self.submit(what)

    # -- what the kernel said ------------------------------------------------------

    @property
    def events(self) -> tuple[AnyEvent, ...]:
        """Every event this run has read, in the order the kernel said them."""
        return tuple(self._events)

    def since(self, at: int) -> tuple[AnyEvent, ...]:
        """The events after the first ``at`` of them — what one step looks through."""
        return tuple(self._events[at:])

    @property
    def cursor(self) -> int:
        """How far the steps have looked — the first event one may yet match.

        A step looks forward from here and moves it past what it matched, so an event that arrived while no
        step was waiting is still found by the next one. That is the difference between "after the previous
        step" and "after the previous match", and only the first one is what a script means (E184).
        """
        return self._cursor

    async def expect(self, wanted: Callable[[AnyEvent], bool], *, patience: float) -> int:
        """Wait for something from the cursor on, and answer where it was found.

        The cursor moves past the match, so two steps expecting two different events see them in the order
        they arrived even when the first one was already over by the time the second step was sent.
        """
        at = await self.wait_for(wanted, at=self._cursor, patience=patience)
        self._cursor = at + 1
        return at

    async def wait_for(self, wanted: Callable[[AnyEvent], bool], *, at: int = 0, patience: float) -> int:
        """Wait until an event from ``at`` on satisfies ``wanted``, and answer where it is.

        ``patience`` is this caller's and not a fact about the kernel, so it is passed in: a run that waited
        forever would turn a plugin that answers nothing into a test that never ends.
        """
        async with asyncio.timeout(patience):
            while True:
                while at < len(self._events):
                    if wanted(self._events[at]):
                        return at
                    at += 1
                if self._reading.done():
                    _ = self._reading.exception()
                    msg = f"the kernel stopped before it said what was expected, after {len(self._events)} events"
                    raise GoneError(msg)
                self._arrived.clear()
                _ = await self._arrived.wait()

    async def finished(self, *, patience: float, turn: int = 1) -> None:
        """Wait for one turn to end, and refuse one that was refused or ended for a reason of its own.

        A refused turn never starts, so waiting for its end would wait for something the kernel never
        promised: its refusal is the answer, and it is said here instead.
        """
        at = await self.wait_for(
            lambda event: (
                (isinstance(event, TurnFinished) and event.turn == turn)
                or (isinstance(event, TurnRefused) and event.session == self.session)
            ),
            patience=patience,
        )
        ended = self._events[at]
        why = ""
        if isinstance(ended, TurnRefused):
            why = f"the kernel refused to run the turn: {ended.reason}"
        elif isinstance(ended, TurnFinished) and (ended.reason or ended.failure):
            why = f"turn {turn} did not finish on its own: {ended.reason or ended.failure}"
        if why:
            raise RuntimeError(why)

    async def close(self) -> None:
        """Stop the kernel, which is what ends the connection the events arrive on."""
        await self._kernel.close()
        with contextlib.suppress(asyncio.CancelledError):
            await self._reading

    async def _read(self) -> None:
        """Keep every event, tell whoever asked about it, and wake whoever is waiting for one."""
        async for event in self._kernel.client.events():
            self._events.append(event)
            if self.on_event is not None:
                self.on_event(event)
            self._arrived.set()
