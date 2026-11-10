"""A conversation, and the address of one of its turns.

A session is a component so that the things the design asks of it are answers rather than bookkeeping
of our own: it comes up and goes down with ``admit`` / ``evict``, its ``drain`` is where the turns it
still has are told to stop, and ``in_transit`` / ``state`` say what it is doing without a flag
somewhere in the CLI. Its requirements are read one level up the address — a session at ``ws-a/s1``
takes ``ws-a``'s host — which is what ``answering_from_the_root(..., below=ancestors)`` lays out.

What a session does *not* hold is anything that has to be withdrawn explicitly. A child on a terminal,
an fd, a remote lease belong to the host, because the host is not replaced within a process and a
session is: a session may carry only what stops working when nobody holds it.

A turn is not a component. It is an address and a token, handed to the code that runs it — turns are
far too frequent and too short-lived for a lifecycle, and everything a turn owns is already reachable
through the host and the session it was taken from.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import override

from composat import Component, require

from plyngent.kernel.cancel import CancelToken, Scope

# Not a type-only import: ``require()`` resolves the field's annotation from this module's globals.
from plyngent.kernel.host import Host  # noqa: TC001
from plyngent.kernel.reads import Reads
from plyngent.wire.ask import AskSite
from plyngent.wire.event import TurnKind
from plyngent.wire.scope import CancelReason, CancelScope


@dataclass(frozen=True, slots=True)
class TurnCtx:
    """One turn's address: the session and host it works through, and where its cancellation lives.

    The host is captured when the turn begins rather than read again later: ``Session.host`` is a
    live read that raises once the assembly is down, and a turn that already started still has to
    settle its results into the history after the session has begun to leave.

    ``begin_after`` is the entry this turn continues, and ``None`` is a root. It is recorded here and
    carried on ``TurnStarted``; which branch a frontend is looking at is the frontend's own state, so
    nothing here answers "where am I".
    """

    session: Session
    host: Host
    turn_id: int
    round_id: int
    cancel: CancelToken
    kind: TurnKind = TurnKind.CHAT
    begin_after: str | None = None
    since: float = 0.0
    #: What this turn has shown with line numbers, so ``edit_lineno`` can tell a range it may change
    #: from a range nobody read. It is mutable and shared by every round of the turn — the rounds are
    #: one turn, and "read in round one" is not unread in round two.
    reads: Reads = field(default_factory=Reads)

    def turn_address(self) -> CancelScope:
        """Where a whole turn is cancelled — the first segment every other address here shares."""
        return CancelScope((str(self.turn_id),))

    def address(self, *segments: str) -> CancelScope:
        """An address under this turn's current round."""
        return self.turn_address().child(f"r{self.round_id}", *segments)

    def scope(self, *segments: str) -> Scope:
        """The cancellation handle for one piece of work under this turn's current round."""
        return self.cancel.scope(self.address(*segments))

    def site(self, *segments: str) -> AskSite:
        """Where a question asked from here belongs, as the value the human channel takes.

        The caller names its own place — a tool passes the segment it is working under — so the
        transport never has to guess which turn a question came from, and no ambient state has to
        carry it.
        """
        return AskSite(session=self.session.session_id, turn=self.turn_id, scope=str(self.address(*segments)))

    def round_scope(self) -> Scope:
        """The cancellation handle for the round itself."""
        return self.cancel.scope(self.turn_address().child(f"r{self.round_id}"))

    def next_round(self) -> TurnCtx:
        """The same turn, one round on — a new value, so a round cannot be advanced in place."""
        return TurnCtx(
            session=self.session,
            host=self.host,
            turn_id=self.turn_id,
            round_id=self.round_id + 1,
            cancel=self.cancel,
            kind=self.kind,
            begin_after=self.begin_after,
            since=self.since,
            reads=self.reads,
        )

    def end(self) -> None:
        """Close this turn: its token is kept by nothing afterwards."""
        self.session.end_turn(self)


@dataclass(frozen=True, slots=True)
class TurnClaim:
    """One live turn as anything outside may see it: what it is, and where it started.

    ``doing`` is deliberately absent: a sentence about what a turn is *currently* doing would be
    prose that goes stale between the write and the read. Addresses and states are answers; a status
    string is a guess with a clock on it.
    """

    turn: int
    kind: TurnKind
    begin_after: str | None
    since: float


@dataclass(frozen=True, slots=True)
class SessionStatus:
    """What one session is doing, answered in a single read.

    ``heads`` and ``roots`` are entry ids: the entries nothing continues from, and the entries with no
    parent. Both are empty for a session whose journal holds nothing, which is the truth rather than a
    placeholder — a session with no entries has neither. ``seq`` is not here either: it is the
    transport's watermark and belongs to the link that stamped it.
    """

    session: str
    turns: tuple[TurnClaim, ...]
    heads: tuple[str, ...]
    roots: tuple[str, ...]


class Session(Component):
    """One conversation: the turns it has, and the cancellation they are addressed through.

    ``session_id`` is the tail of the qualifier the session is declared at (``ws-a/s1``); it is the
    same string a ``Cancel`` message carries as the first segment of a scope.

    A session keeps no "current branch": every turn records the entry it continues (``begin_after``),
    and nothing here decides what a frontend is looking at — so two frontends may read two branches of
    the same conversation and neither is wrong.
    """

    host: Host = require()
    session_id: str
    _turns: dict[int, TurnCtx]

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._turns = {}

    # -- turns ---------------------------------------------------------------------

    async def begin_turn(self, *, begin_after: str | None = None, kind: TurnKind = TurnKind.CHAT) -> TurnCtx:
        """Address a new turn and record it as live.

        Reading ``self.host`` here is the one place the session touches its host for a turn: a
        requirement answers at whatever generation is serving, so the turn works through the host that
        was up when it started rather than through a field that may be gone by the time it settles.

        ``begin_after`` is the entry the turn continues, and the only rule is that such an entry must
        exist — the turn is otherwise free to grow from anywhere, which is what makes branching a
        frontend's decision rather than a command. The id itself is the journal's to allocate, because
        a stored log has to resume the count where the last process left it (D17).
        """
        host = self.host
        if begin_after is not None and (await host.trail.entry(self.session_id, begin_after)) is None:
            msg = (
                f"no entry {begin_after!r} in session {self.session_id}: a turn can only continue an entry that exists"
            )
            raise ValueError(msg)
        turn = TurnCtx(
            session=self,
            host=host,
            turn_id=await host.trail.next_turn(self.session_id),
            round_id=0,
            cancel=CancelToken(),
            kind=kind,
            begin_after=begin_after,
            since=time.monotonic(),
        )
        self._turns[turn.turn_id] = turn
        return turn

    def end_turn(self, turn: TurnCtx) -> None:
        """Forget a turn; a scope that names it afterwards fires nothing."""
        _ = self._turns.pop(turn.turn_id, None)

    def turns(self) -> tuple[TurnCtx, ...]:
        """The turns still live, oldest first."""
        return tuple(self._turns[key] for key in sorted(self._turns))

    async def status(self) -> SessionStatus:
        """What this session is doing, in one read: its live turns and where its history stands.

        This is the answer a frontend used to assemble from its own "compacting", "waiting for
        approval", "a turn is running" flags — and the reason those flags can go: one read either says
        a turn is live or does not, and the journal says where the conversation can be continued.
        """
        shape = await self.host.trail.shape(self.session_id)
        return SessionStatus(
            session=self.session_id,
            turns=tuple(
                TurnClaim(turn=turn.turn_id, kind=turn.kind, begin_after=turn.begin_after, since=turn.since)
                for turn in self.turns()
            ),
            heads=shape.heads,
            roots=shape.roots,
        )

    # -- cancellation --------------------------------------------------------------

    def cancel(self, scope: CancelScope | None, reason: CancelReason, detail: str = "") -> bool:
        """Cancel ``scope``, or the whole session when it is ``None``; ``False`` when nothing moved.

        The turn is the first segment because that is what makes the address portable: a consumer that
        knows only a session and a turn id can name any scope under it. A scope naming no turn of this
        session fires nothing rather than raising — a cancel that arrived too late is not an error, and
        a consumer that guessed a turn id is told the same thing.

        Session-wide means every live turn, and only the live ones: a turn that already ended has
        nothing left to stop, and firing at it would say it was cancelled rather than finished.
        """
        if scope is None:
            moved = False
            for turn in self.turns():
                moved = turn.cancel.fire(turn.turn_address(), reason, detail) or moved
            return moved
        try:
            turn_id = int(scope.path[0])
        except ValueError:
            return False
        turn = self._turns.get(turn_id)
        if turn is None:
            return False
        return turn.cancel.fire(scope, reason, detail)

    # -- lifecycle -----------------------------------------------------------------

    @override
    async def drain(self) -> None:
        """Tell every live turn the session is going down, then wait for their work to report.

        The order matters and cannot be moved: ``disable`` runs after this hook, so a turn cannot be
        asked to stop later — the reason has to arrive here, and the settle that lets a cancelled tool
        write its result into the history has to happen here too.

        The terminals this session opened go back to the host *after* that settle, and through the
        host, because a child on a terminal outlives no host but does outlive the turn that started it.
        A slot nobody returns is a slot the next session with this id inherits as already spent.
        """
        turns = self.turns()
        for turn in turns:
            _ = turn.cancel.fire(turn.turn_address(), CancelReason.LEAVING)
        for turn in turns:
            _ = await turn.cancel.settle(turn.turn_address())
        _ = await self.host.pty.close_owned(self.session_id)

    @override
    async def unload(self) -> None:
        """Let go of the turns: nothing may be cancelled through a session that is down."""
        self._turns.clear()
