"""The kernel's half of one connection: what a client asked for, done, and what happened said back.

A connection is a client's intent and the kernel's answer, one command at a time. This is the answer
side, and it is the only place that holds the two things a frontend must not: the assembly, and the
right to admit a session into it. Everything it needs to *say* goes out through the kernel face of
the link, so no part of this file knows there is a terminal on the other end.

Two rules shape it:

* **A command is an intent, not a call.** ``Open`` admits a session, ``Close`` evicts it, ``Submit``
  starts a turn — the dispatcher decides, and the client learns the outcome from the event lane. There
  is no reply to correlate (R2.1 D5), so a command whose meaning has not landed yet is *accepted and
  silent* rather than refused with an invented event.
* **The human belongs to the connection.** Attaching is a ``replace`` of ``Host.human`` with a
  :class:`~plyngent.kernel.link.LinkHuman` over this link, and the end of the command lane is what
  detaches it — a client that is gone cannot answer, so the host goes back to "nobody to ask", which
  answers "no" rather than hanging.

What is not here yet, and is honest about it: turns have no runner (§R5), so ``Submit`` says so with a
``TurnRefused`` instead of pretending a turn ran; and ``Compact`` / ``Prune`` / ``Resume`` /
``Export`` / ``SetRoute`` have no meaning until the history store and the route component exist, so
they are accepted and do nothing. Those are the places this file grows next, not places it lies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from composat import Declaration
from composat.errors import KeyInTransitError

from plyngent.kernel.events import HumanChannel
from plyngent.kernel.link import LinkHuman
from plyngent.kernel.session import Session
from plyngent.wire.command import Cancel, Close, ListSessions, Open, Submit
from plyngent.wire.event import KernelLeaving, SessionsListed, TurnRefused
from plyngent.wire.scope import CancelScope

if TYPE_CHECKING:
    from composat import Assembly

    from plyngent.wire.command import AnyCommand
    from plyngent.wire.link import KernelSide


class Serving:
    """One kernel, one connection, one command loop.

    ``host`` is the host's qualifier in the assembly, and also the prefix a session is admitted under
    (``ws/s1``), because that is what makes a session's requirements answer one level up.
    """

    assembly: Assembly
    host: str
    link: KernelSide
    _human: Declaration
    _detached: bool
    _stopped: bool

    def __init__(self, assembly: Assembly, *, host: str, link: KernelSide) -> None:
        self.assembly = assembly
        self.host = host
        self.link = link
        self._human = Declaration(HumanChannel, instance=host)
        self._detached = False
        self._stopped = False

    # -- the loop ------------------------------------------------------------------

    async def run(self) -> None:
        """Be this connection's kernel until the client stops asking, then let the host go.

        The loop ends for one of two reasons and both mean the same thing here: the client closed the
        connection, or :meth:`stop` did. Either way the questions in flight have already been refused
        by the link, and the only thing left to undo is who answers them.
        """
        await self._attach()
        try:
            async for command in self.link.commands():
                await self._handle(command)
        finally:
            await self._detach()

    async def stop(self, reason: str = "") -> None:
        """Leave: say so, then close the lane the loop is reading.

        ``KernelLeaving`` first, because D20 makes "the kernel is going down" an event and not an
        exception to catch: a client that sees its event lane end knows why it ended.
        """
        if self._stopped:
            return
        self._stopped = True
        self.link.emit(KernelLeaving(reason=reason))
        self.link.close()

    # -- what a command means ------------------------------------------------------

    async def _handle(self, command: AnyCommand) -> None:
        """Do what one command asks, or say why it cannot be done yet."""
        match command:
            case Open(session=session, seed_from=seed):
                await self._open(session, seed)
            case Close(session=session):
                await self._close(session)
            case ListSessions():
                self.link.emit(SessionsListed(sessions=self._served()))
            case Cancel(session=session, scope=scope, reason=reason, detail=detail):
                found = self._session(session)
                if found is not None:
                    _ = found.cancel(CancelScope.parse(scope) if scope is not None else None, reason, detail)
            case Submit(session=session):
                self.link.emit(TurnRefused(session=session, reason=NO_RUNNER, holder=0))
            case _:
                # SetRoute / Compact / Prune / Resume / Export: accepted, and nothing to do until the
                # route component and the history store exist. Silence is the contract's answer for a
                # command with no outcome to report (D5), not an oversight.
                return

    async def _open(self, session: str, seed_from: tuple[str, str] | None) -> None:
        """Start serving a session, unless it is already served.

        ``Open`` is idempotent because a frontend that reattaches has the same session in mind; what it
        may not have is a second copy of it. Seeding is the one form that needs entries to point at, so
        it waits for the forest.
        """
        if seed_from is not None:
            self.link.emit(TurnRefused(session=session, reason=NO_FOREST, holder=0))
            return
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        if declaration.key in declared:
            return
        _ = await self.assembly.admit(declaration)

    async def _close(self, session: str) -> None:
        """Stop serving a session; its PTYs go back to the host on the way down."""
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        if declaration.key not in declared:
            return
        _ = await self.assembly.evict(declaration.key)

    # -- who answers questions -----------------------------------------------------

    async def _attach(self) -> None:
        """Make this connection the one that answers, or leave the host as it was.

        A second connection while the first is still attaching is refused by the framework
        (``KeyInTransitError``) and that is the right answer for *it*: one approval owner, no election
        (D19). This connection then simply gets no questions, which is what "nobody to ask" already
        means.
        """
        try:
            _ = await self.assembly.replace(
                self._human.key, Declaration(HumanChannel, instance=self.host, args=(LinkHuman(self.link),))
            )
        except KeyInTransitError:
            return

    async def _detach(self) -> None:
        """Put the host back to "nobody to ask", because this client cannot answer any more."""
        if self._detached:
            return
        self._detached = True
        try:
            _ = await self.assembly.replace(self._human.key, Declaration(HumanChannel, instance=self.host))
        except KeyInTransitError:
            # Somebody else is mid-attach; their generation is not ours to withdraw.
            return

    # -- the addresses -------------------------------------------------------------

    def _declaration(self, session: str) -> Declaration:
        """How a session with this id is declared: under the host, with the id as its tail."""
        return Declaration(Session, instance=f"{self.host}/{session}", kwargs={"session_id": session})

    def _session(self, session: str) -> Session | None:
        """The live session with this id, or ``None``.

        A cancel naming a session this kernel does not serve is not an error: it is a message that
        arrived too late or from a frontend that guessed, and both are answered by doing nothing.
        """
        found = self.assembly.instance(self._declaration(session).key)
        return found if isinstance(found, Session) else None

    def _served(self) -> tuple[str, ...]:
        """The session ids this kernel is serving, in the order they were admitted."""
        identity = Declaration(Session).identity
        declared = self.assembly.keys()
        return tuple(key.instance.rpartition("/")[2] for key in declared if key.identity == identity)


#: Why a ``Submit`` did not start a turn: there is no turn runner yet, and saying so beats pretending.
NO_RUNNER = "this kernel cannot run turns yet"

#: Why an ``Open`` could not seed a new session: there is no history to copy from yet.
NO_FOREST = "seeding a session needs the history store, which does not exist yet"
