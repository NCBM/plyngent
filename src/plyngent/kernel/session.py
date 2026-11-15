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

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast, override

import msgspec
from composat import Component, require

from plyngent.kernel.cancel import CancelToken, Scope

# Not a type-only import: ``require()`` resolves the field's annotation from this module's globals.
from plyngent.kernel.host import Host  # noqa: TC001
from plyngent.kernel.reads import Reads
from plyngent.kernel.trail import StateChange
from plyngent.wire.ask import AskSite
from plyngent.wire.event import TurnKind
from plyngent.wire.scope import CancelReason, CancelScope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

#: The key a session's tool disclosure is written under. One key in the journal's state vocabulary, and
#: the first one written: what a session has been shown, and the epoch that says which tool set it is.
DISCLOSURE = "disclosure"

#: The key the todo stack is written under. A second key in the same vocabulary, because it is the same
#: kind of thing: something true about this conversation that no model is sent and that comes back when
#: the conversation is opened again (§10.9).
TODOS = "todos"

#: The key the compaction ledger is written under: how many times a summary changed the shape of what
#: the next prompt holds. A third key in the same vocabulary, and the second of the two numbers a
#: provider cache keys on (§10.5): the tools move, and the plan moves, and neither has to move for the
#: other to.
PROMPT_PLAN = "prompt-plan"


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

    What it does keep is what it has been told about itself: the tool groups it has been shown, how
    often its tool set has changed, and the todo stack. Those live here rather than on the turn because
    a turn is too short to be shown anything and too frequent to be the record of it, and they come
    back from the journal in :meth:`load` so that reopening a conversation does not mean asking for the
    same groups again or losing its plan.

    They are kept as **one kind of thing**: a setting, spelled as a key and a whole JSON value, written
    by :meth:`change` and read back by :meth:`state`. The properties beside it are decoders of that
    value rather than fields of their own, so there is one fact in one place and no second copy to keep
    in step with the journal.
    """

    host: Host = require()
    session_id: str
    _turns: dict[int, TurnCtx]
    _state: dict[str, str]
    _lock: asyncio.Lock

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self._turns = {}
        self._state = {}
        self._lock = asyncio.Lock()

    # -- what it knows about itself -------------------------------------------------

    def setting(self, key: str) -> str | None:
        """The value this session last wrote under *key*, or ``None`` when it never wrote one.

        A read and not a fold: the fold happened once, in :meth:`load`, and what it produced is this
        process's answer until a write replaces it. That is why reading it is synchronous — there is
        nothing to wait for, and a tool that had to await its own setting would be a tool racing itself.

        It is not called ``state`` because :class:`~composat.Component` already has that name, and a
        session's own facts are not the component's lifecycle.
        """
        return self._state.get(key)

    async def change[T](self, key: str, mutate: Callable[[str | None], tuple[str | None, T]]) -> T:
        """Read one setting, change it, write it back, and answer what *mutate* answered.

        The read-modify-write happens under this session's lock, and the answer is returned from
        inside it because that is the only place the value it was computed from is true: two calls in
        one answer that both read the todo stack outside a lock would each write down the group they
        saw, and one of them would be gone.

        ``None`` for the new value means the call changed nothing; so does a value equal to the one
        already written. Neither writes an entry, because a setting that did not move must not make
        the history say something happened — and because a journal is read by people.
        """
        async with self._lock:
            before = self._state.get(key)
            after, answer = mutate(before)
            if after is None or after == before:
                return answer
            _ = await self.host.trail.append(self.session_id, StateChange(key=key, value=after))
            self._state[key] = after
            return answer

    # -- what it has been shown ----------------------------------------------------

    @property
    def disclosed(self) -> frozenset[str]:
        """The tool groups this session has been shown, beside the residents it always has (§10.4)."""
        return _disclosure(self._state.get(DISCLOSURE))[0]

    @property
    def toolset_epoch(self) -> int:
        """How many times this session's tool set has changed.

        A number and not a hash: what a provider cache needs is a short key that moves when the tools
        block moves, and what we need is to know the miss was ours (§10.5). It counts the changes we
        made on purpose — a disclosure, and later a relist or a client attaching — because a change
        we cannot name is a change we cannot account for.
        """
        return _disclosure(self._state.get(DISCLOSURE))[1]

    async def disclose(self, groups: Iterable[str]) -> frozenset[str]:
        """Show this session these groups, and answer which of them it had not seen.

        The whole set is written rather than the difference, so reading it back is one value and not a
        replay: a fold that had to apply deltas in order would be a second projector to keep in step.
        Nothing is written when nothing is new, which is what makes two calls asking for one group in
        one answer cost a single epoch — the second call is answered from the state the first one left.
        """
        asked = frozenset(groups)

        def mutate(before: str | None) -> tuple[str | None, frozenset[str]]:
            shown, epoch = _disclosure(before)
            fresh = asked - shown
            if not fresh:
                return None, frozenset()
            return _disclosure_value(shown | fresh, epoch + 1), fresh

        return await self.change(DISCLOSURE, mutate)

    async def notice_tools(self) -> int:
        """Say this session's tool set moved without it asking, and answer the number it stands at.

        Two things move a session's tool set besides its own asking, and neither is something it can
        write itself: a client attaching its tools and a client leaving with them. Both change the
        block the next request carries, so both have to change the number a provider cache keys on —
        otherwise the prefix moves quietly, and a miss we caused is a miss we cannot account for
        (§10.5, §10.8).

        The disclosure's groups are rewritten unchanged: what moved is the set, not what was asked for.
        """

        def mutate(before: str | None) -> tuple[str | None, int]:
            shown, epoch = _disclosure(before)
            return _disclosure_value(shown, epoch + 1), epoch + 1

        return await self.change(DISCLOSURE, mutate)

    # -- what the prompt plan is made of -------------------------------------------

    @property
    def prompt_plan_epoch(self) -> int:
        """How many times what the next prompt holds has changed — every compaction, so far.

        The second of the two numbers a provider cache keys on (§10.5), and a separate one because the
        two move for different reasons: a disclosure changes the tools block, and a compaction changes
        the conversation under it. Keeping them apart is what makes a cache miss attributable to one of
        them instead of to "something happened".
        """
        return _epoch(self._state.get(PROMPT_PLAN))

    async def note_compaction(self) -> int:
        """Say a summary just changed what the next prompt holds, and answer the number it stands at.

        Written by whoever wrote the summary, because the count lives beside the history it counts: a
        session reopened after a compaction has to know the plan moved, or the first request of the new
        process would claim a prefix no provider ever answered for.
        """

        def mutate(before: str | None) -> tuple[str | None, int]:
            epoch = _epoch(before) + 1
            return msgspec.json.encode({"epoch": epoch}).decode(), epoch

        return await self.change(PROMPT_PLAN, mutate)

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
    async def load(self) -> None:
        """Read back what this session already knows about itself, before its first turn.

        A session is admitted fresh every time a client opens it, so anything it learned in an earlier
        process has to come out of the journal: which tool groups it was shown, how many times its tool
        set has changed, and what its plan was. The fold is the journal's — the last value written per
        key — so this is one read and not a replay of the settings.

        A stored value this build cannot read is skipped rather than fatal, and skipping it is each
        decoder's job: a conversation that cannot be opened at all is worse than one that forgot a
        setting.
        """
        self._state.update(await self.host.trail.state_of(self.session_id))

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


def _disclosure(value: str | None) -> tuple[frozenset[str], int]:
    """The groups a disclosure setting holds, and the epoch it was written with.

    A value this build cannot read answers "nothing has been shown" rather than raising, and it is read
    here rather than in :meth:`Session.load` so that a stored value a later build cannot parse costs a
    forgotten setting and not an unopenable conversation.
    """
    if not value:
        return frozenset(), 0
    try:
        record = msgspec.json.decode(value)
    except msgspec.DecodeError:
        return frozenset(), 0
    if not isinstance(record, dict):
        return frozenset(), 0
    stored = cast("dict[str, object]", record)
    groups = stored.get("groups")
    shown = (
        frozenset(str(name) for name in cast("list[object]", groups)) if isinstance(groups, list) else frozenset[str]()
    )
    epoch = stored.get("toolset_epoch")
    return shown, epoch if isinstance(epoch, int) else len(shown)


def _disclosure_value(shown: frozenset[str], epoch: int) -> str:
    """One disclosure setting as the text written under its key — the whole set, not a difference."""
    return msgspec.json.encode({"groups": sorted(shown), "toolset_epoch": epoch}).decode()


def _epoch(value: str | None) -> int:
    """The number a counter setting holds; a value this build cannot read is a counter at zero.

    Read here rather than in :meth:`Session.load` for the same reason a disclosure is: a stored value a
    later build cannot parse costs a forgotten count and not an unopenable conversation.
    """
    if not value:
        return 0
    try:
        record = msgspec.json.decode(value)
    except msgspec.DecodeError:
        return 0
    if not isinstance(record, dict):
        return 0
    epoch = cast("dict[str, object]", record).get("epoch")
    return epoch if isinstance(epoch, int) else 0
