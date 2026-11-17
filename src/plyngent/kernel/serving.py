# As in `render`: the `case _ as never: assert_never(never)` at the end of the command match is the
# assertion that a new command must be handled here. pyright calls that pattern redundant because it has
# itself proved the union is covered. The redundancy *is* the assertion.
# pyright: reportUnnecessaryComparison=false

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

What the commands do not yet have: ``Resume`` answers with what it is missing, because reattaching a
session to another journal needs a *second* store and this build keeps one per workspace — so a
conversation is resumed by opening it, and ``Open(seed_from=)`` is the other thing that reads one
conversation into another. Nothing is accepted in silence any more: each command is answered, even where
the answer is a refusal. ``Compact``, ``Prune`` and ``Export`` are not turns — the first two write one
bookkeeping row where the conversation ends, the third writes a document and moves nothing — so none of
them takes the session's single turn slot.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, assert_never

from composat import Declaration
from composat.errors import KeyInTransitError

from plyngent.kernel.catalog import BUILTIN, RESIDENT, ToolCatalog
from plyngent.kernel.client import ClientOffers, Offers, specs_from
from plyngent.kernel.documents import MAX_TEXT_CHARS
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.executor import ToolExecutor
from plyngent.kernel.history import Unreadable, Unwilling
from plyngent.kernel.host import Host
from plyngent.kernel.link import LinkHuman
from plyngent.kernel.notices import ELIDED, elided, elision_summary
from plyngent.kernel.paths import display, normalize
from plyngent.kernel.session import Session
from plyngent.kernel.trail import BY_PERSON, TURN_ENDED_TAG, Summary, Withheld, kind_of, preview, transcript
from plyngent.kernel.turn import TurnEnd, TurnRunner
from plyngent.wire.command import (
    Cancel,
    Close,
    Compact,
    Export,
    Goal,
    ListSessions,
    Offer,
    Open,
    Prune,
    Resume,
    Rollback,
    SetRoute,
    Submit,
    Ungrant,
)
from plyngent.wire.event import (
    Compacted,
    CompactRefused,
    DocumentStored,
    Entry,
    ExportRefused,
    GoalMode,
    GoalRefused,
    GoalSet,
    GrantRevoked,
    KernelLeaving,
    ListedSession,
    OfferRefused,
    Pruned,
    PruneRefused,
    Refusal,
    ResumeRefused,
    RollbackRefused,
    RolledBack,
    Routed,
    RouteRefused,
    Seeded,
    SeedRefused,
    SessionsListed,
    TurnKind,
    TurnRefused,
)
from plyngent.wire.manifest import CLIENT_OFFER
from plyngent.wire.query import (
    Forest,
    Grants,
    GrantTable,
    LiveTurn,
    LiveTurns,
    MissingRoot,
    Reached,
    ShadowedSkill,
    SkillRow,
    Skills,
    SkillTable,
    Trajectory,
    Turns,
)
from plyngent.wire.scope import CancelReason, CancelScope

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from composat import Assembly

    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.toolspec import ToolSpec
    from plyngent.kernel.trail import Trail
    from plyngent.wire.command import AnyCommand, OfferedTool
    from plyngent.wire.link import KernelSide
    from plyngent.wire.query import AnyQuery, AnyReply


class Serving:
    """One kernel, one connection, one command loop.

    ``host`` is the host's qualifier in the assembly, and also the prefix a session is admitted under
    (``ws/s1``), because that is what makes a session's requirements answer one level up.

    It is handed the tool set as a tuple of declarations rather than a catalog, because it is the one
    place that adds to it: a client offering its tools appends to this table and rebuilds the catalog
    and the executor around the new set (§10.8). A turn that already started holds the table it began
    with, which is what makes "the tools do not change inside a round" free rather than enforced.
    """

    assembly: Assembly
    host: str
    link: KernelSide
    _human: Declaration
    _offers: Declaration
    _answers: bool
    _client_id: str | None
    _detached: bool
    _stopped: bool
    _specs: tuple[ToolSpec, ...]
    _catalog: ToolCatalog
    _executor: ToolExecutor
    _turns: set[asyncio.Task[None]]

    def __init__(
        self, assembly: Assembly, *, host: str, link: KernelSide, specs: tuple[ToolSpec, ...] = BUILTIN
    ) -> None:
        self.assembly = assembly
        self.host = host
        self.link = link
        self._human = Declaration(HumanChannel, instance=host)
        self._offers = Declaration(Offers, instance=host)
        self._answers = False
        self._client_id = None
        self._detached = False
        self._stopped = False
        # What this kernel serves whether or not a client is attached: its own tools, plus whatever was
        # installed for them (§0.9, E26). One table for every session it serves — which of them a
        # session may reach is its own answer (:meth:`ToolCatalog.view`), so what is shared is the
        # registry and not a block.
        self._specs = specs
        self._catalog = ToolCatalog(specs)
        self._executor = ToolExecutor(self._catalog)
        self._turns = set()

    # -- the loop ------------------------------------------------------------------

    async def run(self) -> None:
        """Be this connection's kernel until the client stops asking, then let the host go.

        The loop ends for one of two reasons and both mean the same thing here: the client closed the
        connection, or :meth:`stop` did. Either way the questions in flight have already been refused
        by the link, and the only thing left to undo is who answers them.

        Reads are on their own task, because they are on their own lane and because a read must not wait
        behind a turn: a client asking what it may reach is answered while a model is still thinking,
        and neither question is the other's business.
        """
        await self._attach()
        reading = asyncio.create_task(self._reading())
        try:
            async for command in self.link.commands():
                await self._handle(command)
        finally:
            _ = reading.cancel()
            _ = await asyncio.gather(reading, return_exceptions=True)
            await self._detach()

    async def _reading(self) -> None:
        """Answer every query this client asks, until its lane ends."""
        async for query in self.link.queries():
            _ = self.link.replied(await self._answered(query))

    async def _answered(self, query: AnyQuery) -> AnyReply:
        """What one read is told.

        A read is never refused with an event: its answer is the same kind as the question. The one case
        with nothing to say — a session this kernel does not serve — answers with the empty value, which
        is the truth about a conversation that holds no grants.
        """
        match query:
            case Grants(session=session):
                found = self._session(session)
                held = (
                    ()
                    if found is None
                    else tuple(
                        Reached(path=display(path, root=found.host.workspace.root), mode=mode.name.lower())
                        for path, mode in found.grants.items()
                    )
                )
                return GrantTable(id=query.id, reached=held)
            case Trajectory(session=session):
                entries = await self._trajectory(session)
                return Forest(id=query.id, entries=entries, unfinished=_unfinished(entries))
            case Skills():
                return self._skills(query.id)
            case Turns(session=session):
                return self._live(query.id, session)

    async def _trajectory(self, session: str) -> tuple[Entry, ...]:
        """One conversation's rows, as the forest a person walks: what each one is, and what it continues.

        The journal answers arrival order and parents, and everything else here is said *about* a row
        rather than by it: ``kind`` is the record's own tag, ``preview`` is one line for somebody picking
        a place to continue from, and ``children`` is a count over the same tuple — which is what makes
        "nothing continues this one" visible without a second read (the head is the row whose count is
        zero). A session this kernel is not serving holds no rows here, for the reason a grant table does
        not either: it is a conversation nobody opened.
        """
        found = self._session(session)
        if found is None:
            return ()
        rows = await found.host.trail.entries(session)
        continued: dict[str, int] = {}
        for row in rows:
            if row.begin_after is not None:
                continued[row.begin_after] = continued.get(row.begin_after, 0) + 1
        return tuple(
            Entry(
                id=row.id,
                begin_after=row.begin_after,
                kind=kind_of(row.payload),
                session=row.session,
                turn=row.turn,
                preview=preview(row.payload),
                children=continued.get(row.id, 0),
            )
            for row in rows
        )

    def _live(self, number: int, session: str) -> LiveTurns:
        """The turns this conversation is running, and nothing about where its history stands.

        A session this kernel is not serving runs nothing here, for the reason its grant table is empty:
        it is a conversation nobody opened. What is *not* here is the journal's shape, because that is a
        different question with its own read (``Trajectory``), and an answer that carried both would make
        every client that only wants "is it done yet" ask for the whole forest.
        """
        found = self._session(session)
        if found is None:
            return LiveTurns(id=number)
        return LiveTurns(
            id=number,
            turns=tuple(
                LiveTurn(turn=turn.turn_id, kind=str(turn.kind), begin_after=turn.begin_after) for turn in found.turns()
            ),
        )

    def _skills(self, number: int) -> SkillTable:
        """Every skill this host can see, said as values: the rows, the shadows, and the roots that are not.

        Read out of the discovery rather than kept: it re-reads its roots on every question on purpose,
        so a table cached here would be the one thing in this tree that a person's new skill file could
        not change. A host with no store answers with the empty table, which is the truth about a
        program that was started without one.
        """
        host = self._host()
        if host is None or host.skills is None:
            return SkillTable(id=number)
        store = host.skills
        skills = store.visible()
        return SkillTable(
            id=number,
            skills=tuple(
                SkillRow(
                    name=skill.name,
                    root=skill.root.label,
                    path=str(skill.path),
                    description=skill.description,
                    issues=tuple(skill.issues),
                )
                for skill in skills
            ),
            shadowed=tuple(
                ShadowedSkill(name=name, winner=copies[0].root.label, copied=tuple(c.root.label for c in copies[1:]))
                for name, copies in store.shadowed().items()
            ),
            absent=tuple(MissingRoot(path=str(root.path), label=root.label) for root in store.absent()),
        )

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
            case Open() | Close():
                await self._serving(command)
            case ListSessions():
                self.link.emit(SessionsListed(sessions=await self._served()))
            case Cancel(session=session, scope=scope, reason=reason, detail=detail):
                await self._cancel(session, scope, reason, detail)
            case Submit(session=session, text=text, kind=kind, begin_after=begin_after):
                await self._submit(session, text, kind, begin_after)
            case Offer(tools=tools):
                await self._offer(tools)
            case Compact() | Prune():
                await self._bookkeeping(command)
            case Export(session=session, begin_after=begin_after):
                await self._export(session, begin_after)
            case SetRoute() | Ungrant() | Goal():
                await self._setting(command)
            case Resume() | Rollback():
                await self._past(command)
            case _ as never:
                # Every command a client may send has an answer here, even where the answer is "there is
                # nowhere to resume from": a new command must be a decision at this match, not silence.
                assert_never(never)

    async def _serving(self, command: Open | Close) -> None:
        """The two commands that say which conversations this kernel serves: admit one, or evict one."""
        match command:
            case Open(session=session, seed_from=seed):
                await self._open(session, seed)
            case Close(session=session):
                await self._close(session)

    async def _setting(self, command: SetRoute | Ungrant | Goal) -> None:
        """The commands that move one of a session's own settings rather than touching the world."""
        match command:
            case SetRoute(session=session, route=route):
                await self._route(session, route)
            case Ungrant(session=session, path=path):
                await self._ungrant(session, path)
            case Goal(session=session, mode=mode):
                await self._goal(session, mode)

    async def _goal(self, session: str, mode: GoalMode | None) -> None:
        """Answer where the switch stands, or move it first.

        One refusal, which is the one every setting has: a conversation this kernel does not serve. What
        the answer says is the mode the session now *has*, so the same event answers both halves of the
        word — a question and a move — and a person reading a screen is told where the switch is either way.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(GoalRefused(session=session, reason=Refusal.NO_SESSION))
            return
        now = await found.set_goal(mode) if mode is not None else found.mode
        self.link.emit(GoalSet(session=session, mode=now))

    async def _ungrant(self, session: str, path: str) -> None:
        """Take back one path this conversation was allowed, and say which of the two happened.

        The path is resolved the way every path tool resolves one, against the workspace the session's
        host works in, because that is what the tool that granted it wrote down: a grant is a row keyed
        by an absolute path, and "the same path spelled differently" is not a second row.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(GrantRevoked(session=session, path=path, revoked=False))
            return
        target = normalize(path, base=found.host.workspace.root)
        taken = await found.revoke(target)
        self.link.emit(
            GrantRevoked(session=session, path=display(target, root=found.host.workspace.root), revoked=taken)
        )

    async def _cancel(self, session: str, scope: str | None, reason: CancelReason, detail: str) -> None:
        """Stop what one session is doing, as narrowly as the scope says.

        A cancel naming a session this kernel does not serve is a message that arrived too late or from a
        frontend that guessed, and both are answered by doing nothing — which is why the check is here
        rather than a refusal on the wire.
        """
        found = self._session(session)
        if found is not None:
            _ = found.cancel(CancelScope.parse(scope) if scope is not None else None, reason, detail)

    async def _resume(self, session: str, store: str) -> None:
        """Reattach this session to another journal, which this build has none of, so say what is missing.

        The command's shape is R4's: replacing what a session reads from invalidates the history basis of
        anything already running, so nothing about it may happen beside a live turn. What it waits for is
        a *second* store — another journal a conversation could be pointed at — and this build keeps one
        per workspace, so "resume" is what opening a conversation already does. Until there is a second
        one the answer is a refusal naming the missing piece, which is more use than the silence this
        command used to keep: a person who typed it learns that the conversation in front of them is
        still the one they were reading.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(ResumeRefused(session=session, reason=Refusal.NO_SESSION))
            return
        self.link.emit(
            ResumeRefused(
                session=session,
                reason=Refusal.UNKNOWN_STORE if store else Refusal.NO_SECOND_STORE,
                store=store,
            )
        )

    async def _past(self, command: Resume | Rollback) -> None:
        """The two commands about what this session and its workspace already were.

        Read together because they are one question asked of two records — a conversation's trajectory
        and a workspace's versions — and both are refused while a turn is live for the same reason.
        """
        match command:
            case Resume(session=session, store=store):
                await self._resume(session, store)
            case Rollback(session=session, version=version, paths=paths, force=force):
                await self._rollback(session, version, paths, force=force)

    async def _rollback(self, session: str, version: str, paths: tuple[str, ...], *, force: bool) -> None:
        """Put the workspace back a version, if this kernel keeps versions and nothing is running.

        A rollback is refused while a turn is live, and for the reason ``Resume`` is: the bytes on disk
        are what that turn's calls are producing, so putting older ones under them would make every
        later result describe a file nobody has. Nothing here asks a person: the command *is* the person,
        and what it touches is what this kernel itself wrote a moment ago.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(RollbackRefused(session=session, reason=Refusal.NO_SESSION))
            return
        store = found.host.history
        if store is None:
            self.link.emit(RollbackRefused(session=session, reason=Refusal.NO_VERSIONS))
            return
        if found.turns():
            self.link.emit(RollbackRefused(session=session, reason=Refusal.TURN_RUNNING))
            return
        try:
            undone = await store.undo(version=version, paths=paths, force=force)
        except (Unreadable, Unwilling) as exc:
            self.link.emit(
                RollbackRefused(
                    session=session,
                    reason=exc.reason,
                    path=getattr(exc, "path", ""),
                    version=getattr(exc, "version", ""),
                    detail=getattr(exc, "detail", ""),
                )
            )
            return
        self.link.emit(
            RolledBack(session=session, revision=undone.revision, target=undone.target, restored=undone.restored)
        )

    async def _bookkeeping(self, command: Compact | Prune) -> None:
        """The two commands that write one row where the conversation ends, and nothing else.

        Neither is a turn, and neither is exclusive: a row does not take the session's single turn slot, so
        a compaction or a prune can be written while a chat is running — which is the same rule the
        summary's arrival already rests on. They are read apart from the rest because they are the same
        shape: a command whose whole outcome is one entry plus one event.
        """
        match command:
            case Compact(session=session, begin_after=begin_after, replaces=replaces, text=text):
                await self._compact(session, begin_after, replaces, text)
            case Prune(session=session, targets=targets, reason=reason):
                await self._prune(session, targets, reason)

    async def _open(self, session: str, seed_from: tuple[str, str] | None) -> None:
        """Start serving a session, unless it is already served — and, if asked, from another's text.

        ``Open`` is idempotent because a frontend that reattaches has the same session in mind; what it
        may not have is a second copy of it. Seeding is the one form that needs entries to point at, so it
        is done first: a copy that refused has written nothing, and the session is not admitted either,
        because an empty conversation would be a worse answer than the reason.

        The opening answers with the route this session stands at, whether or not it was already
        served: a client that has just attached is told where it is rather than having to ask, and the
        answer is the same one ``SetRoute`` gives so a client reads both lines with one rule.
        """
        if seed_from is not None and not await self._seed(session, *seed_from):
            return
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        if declaration.key not in declared:
            _ = await self.assembly.admit(declaration)
        self._say_route(session)

    async def _seed(self, session: str, source: str, entry: str) -> bool:
        """Fill a conversation with another one's text, and answer whether it happened.

        **The text travels and nothing else.** ``derive`` is the projection — the messages a model would
        have been given from that entry to its root — and each of those is appended here as a row of its
        own, chained in order. No pointer to the other session, no grants, no todo, no route, no name: a
        new conversation inheriting another's approvals would be a privilege escalation dressed as
        convenience, and one that inherited its todo list would be a session somebody else is holding.

        Three refusals, and each is a different sentence: an entry this session does not have, a source
        that projects to nothing, and a conversation this kernel is already serving — where copying again
        would either duplicate what is there or interleave with a turn that is using it.
        """
        if self._session(session) is not None:
            self.link.emit(SeedRefused(session=session, reason=Refusal.ALREADY_OPEN))
            return False
        host = self._host()
        if host is None:
            self.link.emit(SeedRefused(session=session, reason=Refusal.NO_HOST))
            return False
        trail = host.trail
        if await trail.entry(source, entry) is None:
            self.link.emit(SeedRefused(session=session, reason=Refusal.NO_SUCH_SOURCE, source=source, entry=entry))
            return False
        try:
            carried = await trail.derive(source, entry)
        except ValueError:
            self.link.emit(SeedRefused(session=session, reason=Refusal.CHAIN_BROKEN, source=source, entry=entry))
            return False
        if not carried:
            self.link.emit(SeedRefused(session=session, reason=Refusal.NOTHING_TO_CARRY))
            return False
        parent: str | None = None
        for message in carried:
            parent = (await trail.append(session, message, begin_after=parent)).id
        assert parent is not None
        self.link.emit(Seeded(session=session, source=source, from_entry=entry, entry=parent, count=len(carried)))
        return True

    def _say_route(self, session: str) -> None:
        """Say which route answers *session*, resolved the way a route change resolves one."""
        found = self._session(session)
        if found is None:
            return
        asked = found.host.routes.resolve(found.route)
        self.link.emit(
            Routed(
                session=session,
                route="" if asked is None else asked.name,
                epoch=found.route_epoch,
                opening=True,
                context_window=0 if asked is None else asked.caps.context_window,
            )
        )

    async def _close(self, session: str) -> None:
        """Stop serving a session; its PTYs go back to the host on the way down."""
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        if declaration.key not in declared:
            return
        _ = await self.assembly.evict(declaration.key)

    # -- what a client offers ------------------------------------------------------

    async def _offer(self, tools: tuple[OfferedTool, ...]) -> None:
        """Take what this client can run, or say one tool at a time why it was not taken.

        Attaching changes what every session in this kernel is shown, so nothing about it is silent:
        each refusal is an event with the rule it broke, and each session's tool set epoch moves once
        for the whole offer rather than once per tool (§10.8, E44/E46).

        A second ``Offer`` on one connection replaces the first — the client is the same client, and
        what it can run is whatever it last said. An offer that attaches nothing and finds nothing
        attached is the one case that leaves the table alone: there is no change to count.
        """
        if CLIENT_OFFER not in self.link.manifest.caps:
            self.link.emit(OfferRefused(reason=Refusal.NO_CAPABILITY))
            return
        before = self._offers_now()
        client = before.attached.id if before.attached is not None else f"c{before.issued + 1}"
        attached, refused = specs_from(client, tools, taken={spec.name for spec in self._specs})
        for name, reason in refused:
            self.link.emit(OfferRefused(name=name, reason=reason))
        if not attached and (tools or before.attached is None):
            # Nothing was taken: either every tool offered was refused — and a refusal is not a
            # withdrawal — or the client offered nothing and there was nothing to withdraw.
            return
        offered = ClientOffers(
            id=client,
            tools=attached,
            resident=_resident(tools, attached),
            link=self.link,
        )
        issued = before.issued if before.attached is not None else before.issued + 1
        self._client_id = client if attached else None
        _ = await self.assembly.replace(
            self._offers.key,
            Declaration(
                Offers,
                instance=self.host,
                kwargs={"attached": offered if attached else None, "issued": issued},
            ),
        )
        self._retable(attached, offered.resident)
        await self._notice()

    def _retable(self, attached: tuple[ToolSpec, ...], resident: frozenset[str]) -> None:
        """Rebuild the table and the executor around what this kernel now serves.

        A new catalog rather than a change to the old one, because a turn in flight holds the old value:
        replacing it is what makes "a call runs with the tools the round was asked with" true for the
        rounds already running, without anything having to check.
        """
        self._catalog = ToolCatalog((*self._specs, *attached), resident=(*RESIDENT, *resident))
        self._executor = ToolExecutor(self._catalog)

    async def _notice(self) -> None:
        """Tell every session this kernel serves that its tool set moved — once, not once per tool.

        The sessions are the ones admitted, which is the whole list: a session that arrived while the
        client was attached was born with these tools and has nothing to be told, and one that is not
        admitted has no tool set to have changed.
        """
        for session in self._sessions():
            _ = await session.notice_tools()

    def _offers_now(self) -> Offers:
        """What this host currently offers: the component, as it stands."""
        found = self.assembly.instance(self._offers.key)
        return found if isinstance(found, Offers) else Offers()

    # -- what a compaction writes --------------------------------------------------

    async def _route(self, session: str, route: str) -> None:
        """Point one session at a route, or say why nothing moved.

        Two refusals and no third: a session this kernel does not serve, and a name it does not have. An
        empty name is not a refusal — it is "ask whatever this host asks by default", which is a choice
        like any other and the only way back from a route that was set. What the session ends up asking
        is answered resolved, because the name the command carried may have been no name at all.

        Nothing here asks the model anything or waits for a turn: the setting moves, and a turn already
        in flight keeps the route it started with — the change lands at the next request, which is the
        boundary that makes a mid-turn route change legal instead of a race (R7).
        """
        found = self._session(session)
        if found is None:
            self.link.emit(RouteRefused(session=session, reason=Refusal.NO_SESSION))
            return
        known = found.host.routes
        if route and known.named(route) is None:
            self.link.emit(RouteRefused(session=session, reason=Refusal.NO_ROUTE, route=route))
            return
        name, epoch = await found.set_route(route)
        asked = known.resolve(name)
        self.link.emit(
            Routed(
                session=session,
                route="" if asked is None else asked.name,
                epoch=epoch,
                context_window=0 if asked is None else asked.caps.context_window,
            )
        )

    async def _compact(
        self,
        session: str,
        begin_after: str | None,
        replaces: tuple[str, str] | None,
        text: str,
    ) -> None:
        """Write one summary entry, or say why nothing was written.

        **Not a turn**, and that is the whole shape of it: a summary is one row, so a compaction neither
        takes the session's single turn slot nor waits for whoever holds it — a chat and a compaction can
        be in the journal at once, which is what the entry granularity is for (R4).

        ``begin_after=None`` is the ordinary case: the summary is a root, so the conversation continues
        past the whole stretch it stands for. Naming an entry keeps that much of the prefix on the chain
        instead, and naming nothing at all is refused by the journal, whose own sentence is passed on
        rather than rewritten — a parent pointer into nothing is one fact and it is already said once.

        ``replaces`` is bookkeeping rather than a pointer, but it names two entries of *this* conversation
        and nothing else reads it — so a name that is not here would be a claim nothing could ever check
        again. It is refused the way the parent pointer is, before anything is written.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(CompactRefused(session=session, reason=Refusal.NO_SESSION))
            return
        if not text:
            self.link.emit(CompactRefused(session=session, reason=Refusal.NOTHING_TO_KEEP))
            return
        if replaces is not None:
            missing = [named for named in replaces if await found.host.trail.entry(session, named) is None]
            if missing:
                self.link.emit(CompactRefused(session=session, reason=Refusal.NOT_A_SPAN, entry=missing[0]))
                return
        try:
            entry = await found.host.trail.append(
                session, Summary(text=text, replaces=replaces), begin_after=begin_after
            )
        except ValueError:
            self.link.emit(CompactRefused(session=session, reason=Refusal.NO_SUCH_ENTRY, entry=begin_after or ""))
            return
        _ = await found.note_compaction()
        self.link.emit(Compacted(session=session, entry=entry.id, chars=len(text)))

    async def _export(self, session: str, begin_after: str | None) -> None:
        """Write this conversation out as a document, or say why nothing was written.

        A projection, so nothing moves: the messages the next turn would be given are written to the
        document store and answered with the handle that reaches them back. ``begin_after`` names where
        to read from — ``None`` is the end of the conversation — and the kernel writes no files of its
        own, because a path in a command would be a write behind the policy every tool goes through.

        What a person gets is :class:`DocumentStored`, the same event a tool result earns when it is too
        long to send: a document was kept, and its handle is the way back to it.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(ExportRefused(session=session, reason=Refusal.NO_SESSION))
            return
        trail = found.host.trail
        end = begin_after
        if end is None:
            head = await trail.head(session)
            end = None if head is None else head.id
        try:
            messages = await trail.derive(session, end)
        except ValueError:
            self.link.emit(ExportRefused(session=session, reason=Refusal.CHAIN_BROKEN, entry=end or ""))
            return
        if not messages:
            self.link.emit(ExportRefused(session=session, reason=Refusal.NOTHING_TO_EXPORT))
            return
        transcript_text = transcript(messages)
        if len(transcript_text) > MAX_TEXT_CHARS:
            self.link.emit(
                ExportRefused(session=session, reason=Refusal.TOO_BIG, chars=len(transcript_text), limit=MAX_TEXT_CHARS)
            )
            return
        kept = await found.host.documents.put(
            session, transcript_text, summary=f"conversation exported ({len(messages)} messages)"
        )
        self.link.emit(DocumentStored(session=session, document=kept.id, chars=kept.chars))

    async def _prune(self, session: str, targets: tuple[str, ...], reason: str) -> None:
        """Write one entry saying these entries are no longer shown, or say why nothing was written.

        The prune goes where the conversation ends — its parent is the current head — because that is the
        only placement a projection can honour: a turn continues from the entry it is given and walks up
        from there, so a prune written anywhere else would hide nothing from the turn that follows it.
        That also makes the prune the new head, which is what a client's cursor moves to.

        Every target is checked before anything is written: a list naming one entry this session does not
        have writes nothing at all rather than hiding the half of it that did exist, because a prune that
        half happened is a history nobody can account for.

        What the model is told about it is a line the session *owes* rather than a row written here — the
        command may arrive while a turn is in flight, and a message appended inside a round would sit
        between an answer and its results. So the elision is one sentence on the next boundary, and a
        history that stops showing something stops quietly no more (E149).
        """
        found = self._session(session)
        if found is None:
            self.link.emit(PruneRefused(session=session, reason=Refusal.NO_SESSION))
            return
        if not targets:
            self.link.emit(PruneRefused(session=session, reason=Refusal.NOTHING_TO_PRUNE))
            return
        trail = found.host.trail
        for target in targets:
            if await trail.entry(session, target) is None:
                self.link.emit(PruneRefused(session=session, reason=Refusal.NO_SUCH_TARGET, entry=target))
                return
        head = await trail.head(session)
        entry = await trail.append(
            session,
            Withheld(targets=targets, reason=reason, by=BY_PERSON),
            begin_after=None if head is None else head.id,
        )
        # What the model is told is owed rather than written here: a prune may land while a turn is in
        # flight, and a message appended inside a round would sit between an answer and its results.
        found.notices.say(
            ELIDED,
            elided(len(targets), by=BY_PERSON, reason=reason),
            summary=elision_summary(len(targets), by=BY_PERSON),
        )
        self.link.emit(Pruned(session=session, entry=entry.id, targets=targets, reason=reason))

    # -- turns ---------------------------------------------------------------------

    async def _submit(self, session: str, text: str, kind: TurnKind, begin_after: str | None) -> None:
        """Start a turn in the background, or say why no turn started.

        In the background because :meth:`run`'s ``async for`` is the only place a ``Cancel`` is read: a
        turn awaited inline would hold the command lane, and Ctrl+C would arrive as a message nobody was
        listening for — the one bug this whole shape exists to make impossible (§R5, decision B).

        One turn at a time per session. Two would interleave in one journal, and while a forest allows
        that, a frontend's cursor does not: it continues one chain. A compaction *written by a person* is
        not a turn at all — it is one row, written where commands are, so the narrower rule ("a chat and a
        compaction may coexist") is already true without anything being relaxed here. A compaction a
        *model* writes is a turn (``kind=COMPACT``): it is a request, it costs what a request costs, and
        it takes this slot like any other. ``holder`` names the turn in the way, so a client can say
        "waiting on turn 7" rather than invent a message.
        """
        found = self._session(session)
        if found is None:
            self.link.emit(TurnRefused(session=session, reason=Refusal.NO_SESSION, holder=0))
            return
        if not found.host.model.installed:
            self.link.emit(TurnRefused(session=session, reason=Refusal.NO_MODEL, holder=0))
            return
        if kind is TurnKind.ASIDE:
            # Declared in the contract and not implemented here: running it as a chat would write the
            # history an aside exists to leave alone, so it is refused by name instead.
            self.link.emit(TurnRefused(session=session, reason=Refusal.NO_ASIDE, holder=0))
            return
        if kind is TurnKind.GOAL:
            # The one kind nobody may ask for: a goal turn is this kernel saying "the plan is not
            # finished", so a client that sends one is asking for a turn on its own authority (E146).
            self.link.emit(TurnRefused(session=session, reason=Refusal.NO_GOAL, holder=0))
            return
        live = found.turns()
        if live:
            self.link.emit(TurnRefused(session=session, reason=Refusal.BUSY, holder=live[0].turn_id))
            return
        try:
            turned = await found.begin_turn(begin_after=begin_after, kind=kind)
        except ValueError:
            self.link.emit(TurnRefused(session=session, reason=Refusal.NO_SUCH_ENTRY, entry=begin_after or ""))
            return
        task = asyncio.create_task(self._turn(turned, text))
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)

    async def _turn(self, turned: TurnCtx, text: str) -> None:
        """Run one turn to its end, and then whatever the goal switch says about the next one.

        A turn that raised has already announced that it ended badly and has already stopped being the
        session's live turn — the runner's ``finally`` does both — so there is nothing left to close
        here. Letting the exception out of the task instead would only hand it to a task nobody awaits,
        where the loop's default handler would print it onto the terminal the kernel is not allowed to
        write to.

        The chain is here rather than in the runner because this is where a turn is *started*: a goal turn
        is one more turn, with its own address, its own cost and its own end row (§10.4, E146).
        """
        while True:
            runner = TurnRunner(turned, catalog=self._catalog, executor=self._executor)
            try:
                end = await runner.run(text)
            except Exception:  # noqa: BLE001 — a turn's failure is reported as an event, not as a traceback
                return
            following = await self._following(turned, end)
            if following is None:
                return
            turned, text = following, ""

    async def _following(self, turned: TurnCtx, end: TurnEnd) -> TurnCtx | None:
        """The turn a goal asks for, or ``None`` when nothing should be started.

        Four answers, and each of them is a way of not insisting: the switch is off, the plan is finished,
        the turn was stopped or failed (an end other than the empty reason), or it acted on nothing — a
        model that answered without calling anything has nothing a second turn could carry on from. A
        turn already running is the fifth: a person may have typed while this one was finishing, and a
        session has one turn at a time.
        """
        session = turned.session
        if not session.goal or turned.kind is TurnKind.COMPACT:
            return None
        if end.reason or not end.acted or not session.todos.reminder() or session.turns():
            return None
        try:
            return await session.begin_turn(begin_after=end.ends_at, kind=TurnKind.GOAL)
        except ValueError:
            return None

    # -- who answers questions -----------------------------------------------------

    async def _attach(self) -> None:
        """Make this connection the one that answers, or leave the host as it was.

        A second connection while the first is still attaching is refused by the framework
        (``KeyInTransitError``) and that is the right answer for *it*: one approval owner, no election
        (D19). This connection then simply gets no questions, which is what "nobody to ask" already
        means — and it does not put the human back when it leaves either, because the channel it would
        be writing is somebody else's.
        """
        try:
            _ = await self.assembly.replace(
                self._human.key, Declaration(HumanChannel, instance=self.host, args=(LinkHuman(self.link),))
            )
        except KeyInTransitError:
            return
        self._answers = True

    async def _detach(self) -> None:
        """Put the host back to "nobody to ask" and "nothing offered", because this client is gone.

        Both are one step and both are undone here, for the same reason: what the connection was the
        owner of ends with it. The session in the history is told its tool set moved, which is what
        makes the next connection's first request miss the prefix exactly once (§10.8, E46).
        """
        if self._detached:
            return
        self._detached = True
        if self._answers:
            try:
                _ = await self.assembly.replace(self._human.key, Declaration(HumanChannel, instance=self.host))
            except KeyInTransitError:
                # Somebody else is mid-attach; their generation is not ours to withdraw.
                return
        if self._client_id is None:
            return
        current = self._offers_now()
        if current.attached is None or current.attached.id != self._client_id:
            return
        _ = await self.assembly.replace(
            self._offers.key, Declaration(Offers, instance=self.host, kwargs={"issued": current.issued})
        )
        await self._notice()

    # -- the addresses -------------------------------------------------------------

    def _declaration(self, session: str) -> Declaration:
        """How a session with this id is declared: under the host, with the id as its tail."""
        return Declaration(Session, instance=f"{self.host}/{session}", kwargs={"session_id": session})

    def _host(self) -> Host | None:
        """The host this kernel serves under, or ``None`` when the assembly has it down.

        What a read of another conversation needs and a served session does not: seeding writes rows for
        a session this kernel has not admitted, so the journal is reached through the host rather than
        through the session that would have pointed at it.
        """
        found = self.assembly.instance(Declaration(Host, instance=self.host).key)
        return found if isinstance(found, Host) else None

    def _session(self, session: str) -> Session | None:
        """The live session with this id, or ``None``.

        A cancel naming a session this kernel does not serve is not an error: it is a message that
        arrived too late or from a frontend that guessed, and both are answered by doing nothing.
        """
        found = self.assembly.instance(self._declaration(session).key)
        return found if isinstance(found, Session) else None

    def _sessions(self) -> tuple[Session, ...]:
        """Every session this kernel serves, in the order they were admitted.

        The same walk a listing does, and the same identity: a session is what was admitted under one,
        so what this kernel serves is the keys carrying that identity and nothing else.
        """
        identity = Declaration(Session).identity
        declared = self.assembly.keys()
        found = [self.assembly.instance(key) for key in declared if key.identity == identity]
        return tuple(session for session in found if isinstance(session, Session))

    async def _served(self) -> tuple[ListedSession, ...]:
        """Every conversation this journal holds, and where each one ends.

        A listing answers two questions at once because a person has both: which conversations exist, and
        which entry each of them can be continued from. The set is the **journal's**, not this process's:
        a conversation outlives the connection, the process and the turn, and one nobody has opened since
        is exactly the one somebody is looking for.

        What the journal has nothing for is a conversation that has been opened and not yet used, and it
        is listed too, at the end, with no head — "I am here" is a fact about a conversation before
        anything is said in it, and a listing that hid it would hide the session a person just made.
        """
        host = self._host()
        if host is None:
            return ()
        held = await host.trail.sessions()
        listed = [(name, await self._head(host.trail, name)) for name in held]
        listed.extend((session.session_id, "") for session in self._sessions() if session.session_id not in set(held))
        return tuple(ListedSession(session=name, head=head) for name, head in listed)

    @staticmethod
    async def _head(trail: Trail, session: str) -> str:
        """Where one conversation ends: the newest entry a turn may grow from, or nothing.

        The newest entry is always a leaf — a child would have arrived after it — so this is a place a
        turn may grow, not a guess at where a person was looking. Which branch a frontend is reading stays
        the frontend's own state (R2.1 D22).
        """
        entry = await trail.head(session)
        return "" if entry is None else entry.id


def _unfinished(entries: Sequence[Entry]) -> tuple[int, ...]:
    """Which turns of this conversation have rows and no end row of their own.

    A value and not a sentence, and not a verdict either: a turn that is running right now has no end row
    for the same reason a turn a killed process left behind does not, and which of the two it is is a
    different question with its own read (``Turns``). What this answers is the one thing a journal can:
    which turns in *this* log nothing has yet said the end of.
    """
    ended = {entry.turn for entry in entries if entry.kind == TURN_ENDED_TAG}
    return tuple(sorted({entry.turn for entry in entries if entry.turn} - ended))


def _resident(offered: Iterable[OfferedTool], attached: tuple[ToolSpec, ...]) -> frozenset[str]:
    """Which names are resident: the ones the offer says are, and only the ones that were taken.

    A refusal has to be able to keep the flag out of the set: a client that offered a resident tool the
    kernel declined would otherwise leave a name that is always there and never exists.
    """
    names = {spec.name for spec in attached}
    return frozenset(tool.name for tool in offered if tool.resident and tool.name in names)
