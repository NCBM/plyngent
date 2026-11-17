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
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import msgspec
from composat import Component, require

from plyngent.kernel.access import AccessMode, mode_covers, parse_access_mode
from plyngent.kernel.cancel import CancelToken, Scope

# Not a type-only import: ``require()`` resolves the field's annotation from this module's globals.
from plyngent.kernel.host import Host  # noqa: TC001
from plyngent.kernel.notices import Notices
from plyngent.kernel.reads import Reads
from plyngent.kernel.todos import Todos
from plyngent.kernel.trail import Consent, StateChange
from plyngent.wire.ask import Agreed, AskSite, Confirmation
from plyngent.wire.event import GoalMode, TurnKind
from plyngent.wire.scope import CancelReason, CancelScope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from plyngent.wire.ask import AnyAnswer, AnyAsk

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

#: The key the goal switch is written under: whether an unfinished plan keeps this conversation going.
#: A setting like the plan itself, and beside it for the same reason — it is something this conversation
#: is doing, not something one turn is (§10.4, E146).
GOAL = "goal"

#: The key the session's route is written under: which of the host's endpoints answers it, and the epoch
#: that says when that changed. A fourth key in the same vocabulary, and the first one whose value is a
#: *choice* rather than a fact the kernel tracked — which is why the epoch is written beside the name:
#: an attempt records the epoch it ran under, so "which model answered this" is answerable after the fact.
ROUTE = "route"

#: The key the session's own path grants are written under: which paths outside the workspace this
#: conversation may reach, and how far. A fifth key, and the only one whose subject is the world rather
#: than this conversation — it is written because a person said yes once, and it has to outlive the
#: process for that to mean anything (§0.11, E73).
GRANTS = "grants"


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

    async def ask(self, question: AnyAsk, *, tool: str, call: str = "", at: str = "") -> AnyAnswer:
        """Put one question to whoever is outside, write down what came back, and answer it.

        The one place a question leaves the kernel: a tool reaches it through
        :meth:`~plyngent.kernel.env.Env.ask`, the executor through its own approval, and nothing else
        asks — so "this process asked somebody" has one meaning, no second road can grow beside it, and
        no question can be answered without a :class:`~plyngent.kernel.trail.Consent` row.

        ``tool`` names who asked and ``call`` is the call's segment of the turn's address — the same two
        the row carries — while ``at`` is the part of that call the question belongs to (``gate``,
        ``consent``, ``approve``). The place the question is put under is built from ``call`` and ``at``
        and nothing else, so where a frontend shows it and what the journal says about it cannot
        disagree.

        The row is written **after** the answer, because ``by`` is a fact about an answer and not about
        a question: a question nobody got to answer — the turn was stopped while it was out — leaves no
        row, and every row is a row about something a person, a policy or a silence said.
        """
        site = self.site(*(part for part in (call, at) if part))
        answer = await self.host.human.put(question, site=site)
        _ = await self.host.trail.append(
            self.session.session_id,
            Consent(
                turn=self.turn_id,
                call=call,
                tool=tool,
                kind=str(question.__struct_config__.tag),
                severity=str(question.severity) if isinstance(question, Confirmation) else "",
                question=question.prompt,
                by=str(answer.by),
                policy=answer.policy,
                accepted=answer.accepted if isinstance(answer, Agreed) else None,
            ),
            turn=self.turn_id,
        )
        return answer

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


class Session(Component):
    """One conversation: the turns it has, and the cancellation they are addressed through.

    ``session_id`` is the tail of the qualifier the session is declared at (``ws-a/s1``); it is the
    same string a ``Cancel`` message carries as the first segment of a scope.

    A session keeps no "current branch": every turn records the entry it continues (``begin_after``),
    and nothing here decides what a frontend is looking at — so two frontends may read two branches of
    the same conversation and neither is wrong.

    What it does keep is what it has been told about itself: the tool groups it has been shown, how
    often its tool set has changed, the todo stack, and the paths it may reach outside the workspace.
    Those live here rather than on the turn because
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
    #: What this conversation still owes the model: the sentences fired since the last line was written,
    #: and the order they are said in (:class:`~plyngent.kernel.notices.Notices`).
    notices: Notices
    _turns: dict[int, TurnCtx]
    _state: dict[str, str]
    _lock: asyncio.Lock

    def __init__(self, session_id: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.notices = Notices()
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
        """How many times what the next prompt holds has changed — a compaction, an elision, so far.

        The second of the two numbers a provider cache keys on (§10.5), and a separate one because the
        two move for different reasons: a disclosure changes the tools block, and a change to the
        conversation under it — a summary written, a stretch put aside — changes the rest. Keeping them
        apart is what makes a cache miss attributable to one of them instead of to "something happened".
        """
        return _epoch(self._state.get(PROMPT_PLAN))

    async def note_history(self) -> int:
        """Say the conversation under the prompt just changed, and answer the number it stands at.

        Written by whoever changed it — the writer of a summary, the turn that checkpointed — because the
        count lives beside the history it counts: a session reopened afterwards has to know the plan
        moved, or the first request of the new process would claim a prefix no provider ever answered for.
        """

        def mutate(before: str | None) -> tuple[str | None, int]:
            epoch = _epoch(before) + 1
            return msgspec.json.encode({"epoch": epoch}).decode(), epoch

        return await self.change(PROMPT_PLAN, mutate)

    # -- which route answers this session -----------------------------------------

    @property
    def route(self) -> str:
        """The route this session chose, or nothing when it never chose and takes the host's default.

        Read at the boundary of every request and not remembered by a turn: that is what makes a route
        change take effect at the next request rather than needing to refuse anybody (R7's ``SetRoute``).
        """
        return _route(self._state.get(ROUTE))[0]

    @property
    def route_epoch(self) -> int:
        """How many times this session's route has changed — the number an attempt records."""
        return _route(self._state.get(ROUTE))[1]

    async def set_route(self, name: str) -> tuple[str, int]:
        """Point this session at *name*, and answer the name and epoch it now stands at.

        No change writes nothing, and that is not an optimisation: a setting that did not move must not
        make the history say something happened. An empty name is a choice too — it says "ask whatever
        this host asks by default", which is different from the name it happens to resolve to.
        """

        def mutate(before: str | None) -> tuple[str | None, tuple[str, int]]:
            current, epoch = _route(before)
            if current == name:
                return None, (current, epoch)
            return msgspec.json.encode({"name": name, "epoch": epoch + 1}).decode(), (name, epoch + 1)

        return await self.change(ROUTE, mutate)

    # -- whether unfinished work keeps the conversation going -----------------------

    @property
    def mode(self) -> GoalMode:
        """Where the goal switch stands, as the word a person reads it as.

        A setting nobody wrote is ``off``: a conversation does not start driving itself, and "never chose"
        and "chose off" are the same thing to everyone who reads it (unlike a route, where an empty name
        means "whatever the host default is").
        """
        return GoalMode.ON if self._state.get(GOAL) == GoalMode.ON.value else GoalMode.OFF

    @property
    def goal(self) -> bool:
        """Whether a plan that is not finished drives the next turn by itself (§10.4, E146).

        Read after a turn ends and nowhere else: the switch moves whenever a person moves it, and what it
        changes is whether the kernel starts another turn — a decision about *time*, which is the one
        thing a session is allowed to ask of a host.
        """
        return self.mode is GoalMode.ON

    async def set_goal(self, mode: GoalMode) -> GoalMode:
        """Turn the switch on or off, and answer what it is now. A mode it already has writes nothing."""

        def mutate(before: str | None) -> tuple[str | None, GoalMode]:
            if before == mode.value:
                return None, mode
            return mode.value, mode

        return await self.change(GOAL, mutate)

    @property
    def todos(self) -> Todos:
        """The plan this conversation wrote down, as the stack it is.

        The same decoder the todo tools read, in the same place they read it from: one setting, one value,
        and a reader that folds it here rather than in whoever needed it (§10.4, E146).
        """
        return Todos.load(self._state.get(TODOS))

    # -- which paths this conversation may reach -----------------------------------

    @property
    def grants(self) -> Mapping[Path, AccessMode]:
        """The paths outside the workspace this conversation is allowed, and how far.

        Read by every path tool through ``Env.resolve``, which is what makes one "yes" mean the rest of
        the conversation rather than the one call that asked (§0.11, E73). A grant is a gate in front of
        a resolving tool and never a sandbox: what runs inside a granted directory is still judged by the
        command policy, and a path the denylist refuses is refused whatever this says.
        """
        return _grants(self._state.get(GRANTS))

    async def grant(self, path: Path, need: AccessMode) -> bool:
        """Allow *need* on *path* for the rest of this conversation; ``False`` when it already did.

        The whole table is written rather than the one new row, for the same reason the disclosure set
        is: reading it back is one value instead of a replay of differences. A narrower need leaves a
        wider grant alone, so a second "yes" never makes the first one smaller.
        """

        def mutate(before: str | None) -> tuple[str | None, bool]:
            held = _grants(before)
            have = held.get(path)
            if have is not None and mode_covers(have, need):
                return None, False
            held[path] = need if have is None else max(have, need)
            return _grants_value(held), True

        return await self.change(GRANTS, mutate)

    async def revoke(self, path: Path) -> bool:
        """Take back this conversation's permission for *path*; ``False`` when it held nothing.

        The row is removed rather than narrowed, because a grant is one row and "how far" is the whole
        of it: a person who says "not any more" is not asking for a smaller permission. One path and not
        a prefix — a directory's entries are not separate rows, so nothing else is left behind.
        """

        def mutate(before: str | None) -> tuple[str | None, bool]:
            held = _grants(before)
            if path not in held:
                return None, False
            del held[path]
            return _grants_value(held), True

        return await self.change(GRANTS, mutate)

    @property
    def prefix_epoch(self) -> tuple[int, int]:
        """The two numbers a provider's prefix cache is keyed on, in one value (§10.5).

        A pair rather than a sum, because the two move for different reasons and a miss has to be
        attributable to one of them: a disclosure moves the tool block, a compaction moves the
        conversation under it.
        """
        return (self.toolset_epoch, self.prompt_plan_epoch)

    @property
    def cache_key(self) -> str:
        """What a request names its prefix by: this session, and where its prompt stands.

        Short and stable because it is an opaque key a provider hashes beside its own; a session's own
        name is in it because two conversations on one host must not be able to hit each other's cache.
        """
        toolset, plan = self.prefix_epoch
        return f"{self.session_id}:{toolset}:{plan}"

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
        """The turns still live, oldest first.

        The one thing a client cannot work out for itself: a turn is started by a command and ends by an
        event, so "is this conversation still working" is a fact the kernel has and the reader does not
        (``Turns`` is the read that hands it out).
        """
        return tuple(self._turns[key] for key in sorted(self._turns))

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


def _grants(value: str | None) -> dict[Path, AccessMode]:
    """The path grants a setting holds; a value this build cannot read is no grants at all.

    Read here rather than in :meth:`Session.load` for the same reason the other decoders are: a stored
    value a later build cannot parse costs a forgotten grant, and a forgotten grant is refused rather
    than honoured — the direction that is safe to be wrong in.
    """
    if not value:
        return {}
    try:
        record = msgspec.json.decode(value)
    except msgspec.DecodeError:
        return {}
    if not isinstance(record, dict):
        return {}
    stored = cast("dict[str, object]", record).get("paths")
    if not isinstance(stored, dict):
        return {}
    held: dict[Path, AccessMode] = {}
    for path, mode in cast("dict[str, object]", stored).items():
        grant = parse_access_mode(str(mode))
        if grant is not None:
            held[Path(path)] = grant
    return held


def _grants_value(held: Mapping[Path, AccessMode]) -> str:
    """One grants setting as the text written under its key — the whole table, in path order."""
    paths = {str(path): mode.name.lower() for path, mode in sorted(held.items())}
    return msgspec.json.encode({"paths": paths}).decode()


def _route(value: str | None) -> tuple[str, int]:
    """One route setting as the name it chose and the epoch it was written at.

    A value this build cannot read is "never chose", which lands the session on the host's default —
    the same forgiveness every setting here gets, and the one that matters most: a conversation must not
    become unopenable because a model was retired.
    """
    if not value:
        return "", 0
    try:
        record = msgspec.json.decode(value)
    except msgspec.DecodeError:
        return "", 0
    if not isinstance(record, dict):
        return "", 0
    stored = cast("dict[str, object]", record)
    name = stored.get("name")
    epoch = stored.get("epoch")
    return (name if isinstance(name, str) else "", epoch if isinstance(epoch, int) else 0)
