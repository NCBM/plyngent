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

What the commands do not yet have: ``Compact`` / ``Prune`` / ``Resume`` / ``Export`` / ``SetRoute``
have no meaning until the history store and the route component exist, so they are accepted and do
nothing. That is the place this file grows next, not a place it lies.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from composat import Declaration
from composat.errors import KeyInTransitError

from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.executor import ToolExecutor
from plyngent.kernel.link import LinkHuman
from plyngent.kernel.session import Session
from plyngent.kernel.turn import TurnRunner
from plyngent.wire.command import Cancel, Close, ListSessions, Open, Submit
from plyngent.wire.event import KernelLeaving, ListedSession, SessionsListed, TurnKind, TurnRefused
from plyngent.wire.scope import CancelScope

if TYPE_CHECKING:
    from composat import Assembly

    from plyngent.kernel.session import TurnCtx
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
    _catalog: ToolCatalog
    _executor: ToolExecutor
    _turns: set[asyncio.Task[None]]

    def __init__(self, assembly: Assembly, *, host: str, link: KernelSide, catalog: ToolCatalog | None = None) -> None:
        self.assembly = assembly
        self.host = host
        self.link = link
        self._human = Declaration(HumanChannel, instance=host)
        self._detached = False
        self._stopped = False
        # One table of tools for every session this kernel serves: which of them a session may reach is
        # its own answer (:meth:`ToolCatalog.view`), so what is shared is the registry and not a block.
        # The table itself is handed in when a set of tools depends on what the composition root installed
        # (§0.9, E26); nothing else changes, because existence is not a fact a session can move.
        self._catalog = ToolCatalog() if catalog is None else catalog
        self._executor = ToolExecutor(self._catalog)
        self._turns = set()

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
                self.link.emit(SessionsListed(sessions=await self._served()))
            case Cancel(session=session, scope=scope, reason=reason, detail=detail):
                found = self._session(session)
                if found is not None:
                    _ = found.cancel(CancelScope.parse(scope) if scope is not None else None, reason, detail)
            case Submit(session=session, text=text, kind=kind, begin_after=begin_after):
                await self._submit(session, text, kind, begin_after)
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

    # -- turns ---------------------------------------------------------------------

    async def _submit(self, session: str, text: str, kind: TurnKind, begin_after: str | None) -> None:
        """Start a turn in the background, or say why no turn started.

        In the background because :meth:`run`'s ``async for`` is the only place a ``Cancel`` is read: a
        turn awaited inline would hold the command lane, and Ctrl+C would arrive as a message nobody was
        listening for — the one bug this whole shape exists to make impossible (§R5, decision B).

        One turn at a time per session. Two would interleave in one journal, and while a forest allows
        that, a frontend's cursor does not: it continues one chain. The narrower rule — a chat turn and a
        compaction may coexist, because the compaction writes one entry nobody is reading — wants the
        entry granularity R4 brings, and until then refusing is the honest answer: ``holder`` names the
        turn in the way, so a client can say "waiting on turn 7" rather than invent a message.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(TurnRefused(session=session, reason=NO_SESSION, holder=0))
            return
        if not found.host.model.installed:
            self.link.emit(TurnRefused(session=session, reason=NO_MODEL, holder=0))
            return
        live = found.turns()
        if live:
            self.link.emit(TurnRefused(session=session, reason=BUSY, holder=live[0].turn_id))
            return
        try:
            turned = await found.begin_turn(begin_after=begin_after, kind=kind)
        except ValueError as exc:
            self.link.emit(TurnRefused(session=session, reason=str(exc), holder=0))
            return
        task = asyncio.create_task(self._turn(turned, text))
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)

    async def _turn(self, turned: TurnCtx, text: str) -> None:
        """Run one turn to its end.

        A turn that raised has already announced that it ended badly and has already stopped being the
        session's live turn — the runner's ``finally`` does both — so there is nothing left to close
        here. Letting the exception out of the task instead would only hand it to a task nobody awaits,
        where the loop's default handler would print it onto the terminal the kernel is not allowed to
        write to.
        """
        runner = TurnRunner(turned, catalog=self._catalog, executor=self._executor)
        try:
            await runner.run(text)
        except Exception:  # noqa: BLE001 — a turn's failure is reported as an event, not as a traceback
            return

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

    async def _served(self) -> tuple[ListedSession, ...]:
        """The sessions this kernel is serving, in the order they were admitted, and where each ends.

        A listing answers two questions at once because a client has both: which conversations exist, and
        which entry each of them can be continued from. A session with an empty journal has no entry to
        name, and says so with an empty ``head`` rather than by being left out.
        """
        identity = Declaration(Session).identity
        declared = self.assembly.keys()
        names = [key.instance.rpartition("/")[2] for key in declared if key.identity == identity]
        return tuple([ListedSession(session=name, head=await self._head(name)) for name in names])

    async def _head(self, session: str) -> str:
        """Where one session's conversation ends: the newest entry, or nothing for an empty journal.

        The newest entry is always a leaf — a child would have arrived after it — so this is a place a
        turn may grow, not a guess at where a person was looking. Which branch a frontend is reading stays
        the frontend's own state (R2.1 D22).
        """
        found = self._session(session)
        if found is None:
            return ""
        entry = await found.host.trail.head(session)
        return "" if entry is None else entry.id


#: Why a ``Submit`` named a session this kernel is not serving: it was never opened, or it was closed.
NO_SESSION = "no such session here: open one first"

#: Why a ``Submit`` did not start a turn: nothing is installed to ask, and pretending otherwise would be
#: the kernel inventing what the model said.
NO_MODEL = "this kernel has no model installed, so it cannot run a turn"

#: Why a ``Submit`` was refused while a turn was live: one turn at a time, and ``holder`` names the one.
BUSY = "a turn is already running in this session"

#: Why an ``Open`` could not seed a new session: there is no history to copy from yet.
NO_FOREST = "seeding a session needs the history store, which does not exist yet"
