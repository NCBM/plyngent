# As in `render`: the `case _ as never: assert_never(never)` at the end of the command match is the
# assertion that a new command must be handled here. pyright calls that pattern redundant because it has
# itself proved the union is covered. The redundancy *is* the assertion.
# pyright: reportUnnecessaryComparison=false

"""The kernel's half of one tree, as one connection sees it: what was asked there, done, and said back.

A connection with one tree is one of these and no routing; a connection with several has one per tree,
all sharing its link, and which one a command reaches is
:class:`~plyngent.kernel.connection.Connection`'s decision (E154). This is the answer side, and it is
the only place that holds the two things a frontend must not: the assembly, and the right to admit a
session into it. Everything it needs to *say* goes out through the kernel face of the link with this
tree's path stamped on it, so no part of this file knows there is a terminal on the other end — and a
frontend reading two trees can tell their events apart.

Two rules shape it:

* **A command is an intent, not a call.** ``Open`` admits a session, ``Close`` evicts it, ``Submit``
  starts a turn — the dispatcher decides, and the client learns the outcome from the event lane. There
  is no reply to correlate (R2.1 D5), so a command whose meaning has not landed yet is *accepted and
  silent* — nothing invents an event for it.
* **The human belongs to the connection, and the audience to the tree.** A question is asked by a turn,
  and it goes back to the connection that started that turn (``TurnCtx.asker``), so what a frontend
  answers is what its own work asked (E156). What a tree *says* goes to everyone watching it:
  :meth:`~Serving.begin` joins the audience and :meth:`~Serving.end` leaves it, so a second frontend on
  one tree is a second view of the same stream (E154).

What the commands do not yet have: ``Resume`` answers with what it is missing, because reattaching a
session to another journal needs a *second* store and this build keeps one journal per tree — so a
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
from msgspec.structs import replace

from plyngent.kernel.catalog import BUILTIN, RESIDENT, ToolCatalog, ToolTable
from plyngent.kernel.client import ClientOffers, Offers, specs_from
from plyngent.kernel.documents import MAX_TEXT_CHARS
from plyngent.kernel.events import HumanChannel
from plyngent.kernel.executor import ToolExecutor
from plyngent.kernel.history import Unreadable, Unwilling
from plyngent.kernel.host import Host
from plyngent.kernel.link import LinkHuman
from plyngent.kernel.notices import (
    ABSENT,
    ELIDED,
    EMPTY,
    PROVIDERS,
    UNAVAILABLE,
    elided,
    elision_summary,
    providers,
)
from plyngent.kernel.paths import display, normalize
from plyngent.kernel.providing import Tools
from plyngent.kernel.running import TooManyTurns
from plyngent.kernel.session import Session
from plyngent.kernel.trail import (
    BY_PERSON,
    DELETED_TAG,
    TURN_ENDED_TAG,
    Summary,
    Withheld,
    kind_of,
    preview,
    transcript,
)
from plyngent.kernel.trail import Deleted as DeletedRecord
from plyngent.kernel.turn import TurnEnd, TurnRunner
from plyngent.wire.command import (
    Cancel,
    Close,
    Compact,
    Delete,
    Export,
    Goal,
    Inherit,
    ListSessions,
    Offer,
    Open,
    PersistModels,
    Prune,
    Rename,
    Resume,
    Retry,
    Rollback,
    SetModel,
    SetRoute,
    Submit,
    Ungrant,
)
from plyngent.wire.event import (
    Compacted,
    CompactRefused,
    Deleted,
    DocumentStored,
    Entry,
    ExportRefused,
    GoalMode,
    GoalRefused,
    GoalSet,
    GrantRevoked,
    ListedSession,
    ModelSet,
    OfferRefused,
    Persisted,
    PersistProblem,
    PersistRefused,
    Pruned,
    PruneRefused,
    Refusal,
    Renamed,
    ResumeRefused,
    RollbackRefused,
    RolledBack,
    Routed,
    RouteRefused,
    Seeded,
    SeedRefused,
    SessionRefused,
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
    ModelRow,
    Models,
    ModelTable,
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
    from plyngent.wire.command import OfferedTool, TreeCommand
    from plyngent.wire.event import AnyEvent
    from plyngent.wire.link import KernelSide
    from plyngent.wire.query import AnyReply, TreeQuery


#: The turn kinds this kernel starts and a client may not: each is named by its own command, and a
#: ``Submit`` asking for the kind is refused with the reason that belongs to it.
_KERNELS_OWN: dict[TurnKind, Refusal] = {
    TurnKind.GOAL: Refusal.NO_GOAL,
    TurnKind.RETRY: Refusal.NO_RETRY,
}


class Serving:
    """One tree, as one connection sees it: what this client asked for there, and what happened said back.

    A connection with several trees has one of these per tree (``E154``), and they share its link: the
    routing between them is :class:`~plyngent.kernel.connection.Connection`'s, and what a *tree* means by
    a command is here. That is why ``workspace`` exists as well as ``host``: ``host`` is the qualifier
    this tree's components are declared under (``ws``, and ``ws2`` for the second one), while
    ``workspace`` is how the wire names the tree — its path, the same spelling the greeting uses and the
    one stamped onto every event that leaves here.

    It is handed the tool set as a tuple of declarations, because it is the one place that adds to it:
    a client offering its tools appends to this table and rebuilds the catalog and the executor around
    the new set (§10.8). A turn that already started holds the table it began with, which is what makes
    "the tools do not change inside a round" true by construction.
    """

    assembly: Assembly
    host: str
    workspace: str
    link: KernelSide
    _asker: LinkHuman
    _human: Declaration
    _offers: Declaration
    _watching: bool
    _client_id: str | None
    _detached: bool
    _specs: tuple[ToolSpec, ...]
    _attached: tuple[ToolSpec, ...]
    _resident: frozenset[str]
    _epoch: int
    _made: tuple[ToolTable, ToolExecutor] | None
    _made_key: tuple[int, int] | None
    _turns: set[asyncio.Task[None]]
    _recovering: dict[str, asyncio.Task[None]]

    def __init__(
        self,
        assembly: Assembly,
        *,
        host: str,
        workspace: str,
        link: KernelSide,
        specs: tuple[ToolSpec, ...] = BUILTIN,
    ) -> None:
        self.assembly = assembly
        self.host = host
        self.workspace = workspace
        self.link = link
        #: Who a turn this tree starts asks: this connection, always. It is not a component and does not
        #: move, because the answer to "where does this turn's question go" may not depend on who else is
        #: attached (E156) — and it is stamped with this tree, so what it says is countable there.
        self._asker = LinkHuman(link, workspace=workspace)
        self._human = Declaration(HumanChannel, instance=host)
        self._offers = Declaration(Offers, instance=host)
        self._watching = False
        self._client_id = None
        self._detached = False
        # What this kernel serves whether or not a client is attached: its own tools. What a client
        # offers arrives beside them, and what a provider has brought is the process's — the three are
        # joined into one table per round, which is what `table` builds (E175).
        self._specs = specs
        self._attached = ()
        self._resident = frozenset()
        self._epoch = 0
        self._made = None
        self._made_key = None
        self._turns = set()
        self._recovering = {}

    # -- the audience --------------------------------------------------------------

    def begin(self) -> None:
        """Join this tree's audience, so that what happens here reaches this connection.

        Joining is a set and not a ``replace``: a second frontend on the same tree is a second view of one
        conversation and both are told what happens (``E154``). Questions are *not* part of this — a turn
        asks the connection that started it (:attr:`_asker`), so a second watcher never takes a question
        away from the frontend whose own work asked it (E156).
        """
        channel = self._channel()
        if channel is None:
            return
        channel.join(self._asker)
        self._watching = True

    async def end(self) -> None:
        """Leave the audience, and put back what this connection was the owner of.

        What the connection held ends with it, and both halves are undone here for the same reason: it
        is gone, so nobody is there to hear the events or to run the tools it offered. The session in the
        history is told its tool set moved, which is what makes the next connection's first request miss
        the prefix exactly once (§10.8, E46).
        """
        if self._detached:
            return
        self._detached = True
        for session in tuple(self._recovering):
            self._forget_recovery(session)
        if self._watching:
            channel = self._channel()
            if channel is not None:
                channel.leave(self._asker)
            self._watching = False
        if self._client_id is None:
            return
        current = self._offers_now()
        if current.attached is None or current.attached.id != self._client_id:
            return
        _ = await self.assembly.replace(
            self._offers.key, Declaration(Offers, instance=self.host, kwargs={"issued": current.issued})
        )
        await self._notice()

    def _channel(self) -> HumanChannel | None:
        """This tree's audience, or ``None`` when the assembly has the host down."""
        found = self.assembly.instance(self._human.key)
        return found if isinstance(found, HumanChannel) else None

    def _say(self, event: AnyEvent) -> None:
        """Say one event about this tree, named as the greeting names a tree: its path."""
        self.link.emit(event if not self.workspace else replace(event, workspace=self.workspace))

    # -- what a command means ------------------------------------------------------

    async def handle(self, command: TreeCommand) -> None:
        """Do what one command asks of this tree, or say why it cannot be done yet."""
        match command:
            case Open() | Close():
                await self._serving(command)
            case ListSessions():
                self._say(SessionsListed(sessions=await self._served()))
            case Cancel(session=session, scope=scope, reason=reason, detail=detail):
                await self._cancel(session, scope, reason, detail)
            case Submit() | Retry():
                await self._turning(command)
            case Offer(tools=tools):
                await self._offer(tools)
            case Compact() | Prune():
                await self._bookkeeping(command)
            case Export(session=session, begin_after=begin_after):
                await self._export(session, begin_after)
            case SetRoute() | SetModel() | PersistModels() | Ungrant() | Goal() | Rename() | Delete():
                await self._setting(command)
            case Resume() | Rollback():
                await self._past(command)
            case _ as never:
                # Every command a client may send has an answer here, even where the answer is "there is
                # nowhere to resume from": a new command must be a decision at this match, not silence.
                assert_never(never)

    async def answered(self, query: TreeQuery) -> AnyReply:
        """What one read is told.

        A read is never refused with an event: its answer is the same kind as the question. The one case
        with nothing to say — a session this tree is not serving — answers with the empty value, which is
        the truth about a conversation that holds no grants.
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
            case Models(session=session):
                return self._models(query.id, session)
            case Trajectory(session=session):
                entries = await self._trajectory(session)
                return Forest(id=query.id, entries=entries, unfinished=_unfinished(entries))
            case Skills():
                return self._skills(query.id)
            case Turns(session=session):
                return self._live(query.id, session)

    @staticmethod
    def nothing(query: TreeQuery) -> AnyReply:
        """The empty answer of a read's own kind: what a tree nothing has opened is told.

        Not a refusal, because a read has no lane to refuse on — and not a lie either: a tree this kernel
        does not serve holds no conversations, no grants and no turns.
        """
        match query:
            case Grants():
                return GrantTable(id=query.id)
            case Models():
                return ModelTable(id=query.id)
            case Trajectory():
                return Forest(id=query.id)
            case Skills():
                return SkillTable(id=query.id)
            case Turns():
                return LiveTurns(id=query.id)

    async def _trajectory(self, session: str) -> tuple[Entry, ...]:
        """One conversation's rows, as the forest a person walks: what each one is, and what it continues.

        The journal answers arrival order and parents, and everything else here is said *about* a row:
        ``kind`` is the record's own tag, ``preview`` is one line for somebody picking a place to
        continue from, and ``children`` is a count over the same tuple — which is what makes
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

    def _models(self, number: int, session: str) -> ModelTable:
        """Which models the route answering *session* offers, and which one it is asked for.

        Read off the route table and the session's setting every time, because both move: a route may be
        changed and a model chosen, and a client that was told once would print what used to be true. A
        conversation this kernel does not serve answers with the empty table, for the same reason a grant
        table does not: it is a conversation nobody opened.
        """
        found = self._session(session)
        route = None if found is None else found.host.routes.resolve(found.route)
        if found is None or route is None:
            return ModelTable(id=number)
        return ModelTable(
            id=number,
            route=route.name,
            current=found.model,
            models=tuple(
                ModelRow(name=one.name, max_output=one.caps.max_output, context_window=one.caps.context_window)
                for one in route.models
            ),
        )

    def _skills(self, number: int) -> SkillTable:
        """Every skill this host can see, said as values: the rows, the shadows, and the roots that are not.

        Read out of the discovery every time: it re-reads its roots on every question on purpose,
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

    # -- what a command says about this tree ---------------------------------------

    async def _serving(self, command: Open | Close) -> None:
        """The two commands that say which conversations this kernel serves: admit one, or evict one."""
        match command:
            case Open(session=session, seed_from=seed, inherit=inherit):
                await self._open(session, seed, inherit)
            case Close(session=session):
                await self._close(session)

    async def _setting(self, command: SetRoute | SetModel | PersistModels | Ungrant | Goal | Rename | Delete) -> None:
        """The commands that move or read what a session says about itself, and touch nothing outside it."""
        match command:
            case SetRoute(session=session, route=route):
                await self._route(session, route)
            case SetModel(session=session, model=model):
                await self._model(session, model)
            case PersistModels(session=session):
                await self._persisting(session)
            case Ungrant(session=session, path=path):
                await self._ungrant(session, path)
            case Goal(session=session, mode=mode):
                await self._goal(session, mode)
            case Rename(session=session, name=name):
                await self._rename(session, name)
            case Delete(session=session):
                await self._delete(session)

    async def _rename(self, session: str, name: str) -> None:
        """Name a conversation, or take its name away with an empty one.

        One refusal, which is the one every command about a session has: a conversation this kernel does
        not serve. The answer is the name the conversation now *has*, so the same event answers both halves
        of the word — where it is called and what it is called — the way ``Goal`` does for its switch.
        """
        found = self._session(session)
        if found is None:
            self._say(SessionRefused(session=session, reason=Refusal.NO_SESSION))
            return
        self._say(Renamed(session=session, name=await found.set_name(name.strip())))

    async def _delete(self, session: str) -> None:
        """Take a conversation out of the list, and leave every row it holds where it is.

        The row goes where the conversation ends, because that is where a reader of its own log finds what
        was decided about it from the outside. Nothing is un-said, so a turn may still continue from the
        entry above it: what changed is the set of conversations a listing shows.
        """
        found = self._session(session)
        if found is None:
            self._say(SessionRefused(session=session, reason=Refusal.NO_SESSION))
            return
        head = await found.host.trail.head(session)
        _ = await found.host.trail.append(session, DeletedRecord(), begin_after=None if head is None else head.id)
        self._say(Deleted(session=session))

    async def _goal(self, session: str, mode: GoalMode | None) -> None:
        """Answer where the switch stands, or move it first.

        One refusal, which is the one every setting has: a conversation this kernel does not serve. What
        the answer says is the mode the session now *has*, so the same event answers both halves of the
        word — a question and a move — and a person reading a screen is told where the switch is either way.
        """
        found = self._session(session)
        if found is None:
            self._say(GoalRefused(session=session, reason=Refusal.NO_SESSION))
            return
        now = await found.set_goal(mode) if mode is not None else found.mode
        self._say(GoalSet(session=session, mode=now))

    async def _ungrant(self, session: str, path: str) -> None:
        """Take back one path this conversation was allowed, and say which of the two happened.

        The path is resolved the way every path tool resolves one, against the workspace the session's
        host works in, because that is what the tool that granted it wrote down: a grant is a row keyed
        by an absolute path, and "the same path spelled differently" is not a second row.
        """
        found = self._session(session)
        if found is None:
            self._say(GrantRevoked(session=session, path=path, revoked=False))
            return
        target = normalize(path, base=found.host.workspace.root)
        taken = await found.revoke(target)
        self._say(GrantRevoked(session=session, path=display(target, root=found.host.workspace.root), revoked=taken))

    async def _cancel(self, session: str, scope: str | None, reason: CancelReason, detail: str) -> None:
        """Stop what one session is doing, as narrowly as the scope says.

        A cancel naming a session this kernel does not serve is a message that arrived too late or from a
        frontend that guessed, and both are answered by doing nothing — which is why the check is here
        and the wire carries no refusal.
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
            self._say(ResumeRefused(session=session, reason=Refusal.NO_SESSION))
            return
        self._say(
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
            self._say(RollbackRefused(session=session, reason=Refusal.NO_SESSION))
            return
        store = found.host.history
        if store is None:
            self._say(RollbackRefused(session=session, reason=Refusal.NO_VERSIONS))
            return
        if found.turns():
            self._say(RollbackRefused(session=session, reason=Refusal.TURN_RUNNING))
            return
        try:
            undone = await store.undo(version=version, paths=paths, force=force)
        except (Unreadable, Unwilling) as exc:
            self._say(
                RollbackRefused(
                    session=session,
                    reason=exc.reason,
                    path=getattr(exc, "path", ""),
                    version=getattr(exc, "version", ""),
                    detail=getattr(exc, "detail", ""),
                )
            )
            return
        self._say(RolledBack(session=session, revision=undone.revision, target=undone.target, restored=undone.restored))

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

    async def _open(self, session: str, seed_from: tuple[str, str] | None, inherit: Inherit) -> None:
        """Start serving a session, unless it is already served — and, if asked, from another's text.

        ``Open`` is idempotent because a frontend that reattaches has the same session in mind; what it
        may not have is a second copy of it. Seeding is the one form that needs entries to point at, so it
        is done first: a copy that refused has written nothing, and the session is not admitted either,
        because an empty conversation would be a worse answer than the reason.

        ``inherit`` is what the new conversation is told to carry besides the text, and
        ``Inherit.APPROVALS`` is the only value that carries anything: the answers the source was given
        move into the new conversation's table. It happens after the session is admitted because a
        conversation's approvals are the session's own (E156); a source this kernel is not serving has no
        table to carry, which is the same answer as asking about a conversation nobody has open.

        The opening answers with the route this session stands at, whether or not it was already
        served: a client that has just attached is told where it is, with no question to ask, and the
        answer is the same one ``SetRoute`` gives so a client reads both lines with one rule.
        """
        if seed_from is not None and not await self._seed(session, *seed_from):
            return
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        if declaration.key not in declared:
            _ = await self.assembly.admit(declaration)
            # A conversation that is being admitted may have asked for groups in an earlier process; the
            # bringing back starts here so the first round finds it done (E176).
            self._begin_recovery(session)
        if seed_from is not None and inherit is Inherit.APPROVALS:
            self._carry_approvals(session, seed_from[0])
        await self._admitted(session, *(seed_from or ()))
        self._say_route(session)

    async def _admitted(self, session: str, *seeded: str) -> None:
        """Say that this conversation, and whatever it was copied from, has been used (E186).

        Two acts say nothing in the log and are still uses: opening a conversation, and copying its text
        into a new one. What decides "nobody needs this any more" weighs both, so both are recorded here,
        where both are known.
        """
        host = self._host()
        if host is None:
            return
        for one in (session, *seeded):
            await host.trail.admitted(one)

    def _carry_approvals(self, session: str, source: str) -> None:
        """Give *session* every answer *source* was given, and nothing else.

        Two sessions this kernel is not serving have nothing to move between them, so a missing one is
        not a refusal: the conversation that is here simply starts with the questions still to be asked,
        which is the conservative half of the same rule.
        """
        target = self._session(session)
        earlier = self._session(source)
        if target is None or earlier is None:
            return
        target.approvals.inherit(earlier.approvals)

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
            self._say(SeedRefused(session=session, reason=Refusal.ALREADY_OPEN))
            return False
        host = self._host()
        if host is None:
            self._say(SeedRefused(session=session, reason=Refusal.NO_HOST))
            return False
        trail = host.trail
        if await trail.entry(source, entry) is None:
            self._say(SeedRefused(session=session, reason=Refusal.NO_SUCH_SOURCE, source=source, entry=entry))
            return False
        try:
            carried = await trail.derive(source, entry)
        except ValueError:
            self._say(SeedRefused(session=session, reason=Refusal.CHAIN_BROKEN, source=source, entry=entry))
            return False
        if not carried:
            self._say(SeedRefused(session=session, reason=Refusal.NOTHING_TO_CARRY))
            return False
        parent: str | None = None
        for message in carried:
            parent = (await trail.append(session, message, begin_after=parent)).id
        assert parent is not None
        self._say(Seeded(session=session, source=source, from_entry=entry, entry=parent, count=len(carried)))
        return True

    def _say_route(self, session: str) -> None:
        """Say which route answers *session*, and which model it asks of it."""
        found = self._session(session)
        if found is None:
            return
        asked = found.host.routes.resolve(found.route)
        self._say(
            Routed(
                session=session,
                route="" if asked is None else asked.name,
                epoch=found.route_epoch,
                opening=True,
            )
        )
        self._say_model(found, opening=True)

    def _say_model(self, found: Session, *, opening: bool = False) -> None:
        """Say which model *found* asks for, and how much room that model has."""
        asked = found.host.routes.asking(found.route, found.model)
        self._say(
            ModelSet(
                session=found.session_id,
                model="" if asked is None else asked.model,
                epoch=found.model_epoch,
                opening=opening,
                context_window=0 if asked is None else asked.caps.context_window,
            )
        )

    async def _close(self, session: str) -> None:
        """Stop serving a session: its PTYs go back to the host, and its scratch goes with it.

        Scratch outlives a turn and not a conversation (E152), so this is where it is reclaimed: the
        directories were made for this session and no other conversation could reach them, and keeping
        them until the host went down would hold the slots of sessions nobody has open any more.
        """
        declaration = self._declaration(session)
        declared = self.assembly.keys()
        self._forget_recovery(session)
        if declaration.key not in declared:
            return
        found = self._session(session)
        workspace = None if found is None else found.host.workspace
        _ = await self.assembly.evict(declaration.key)
        if workspace is not None:
            _ = workspace.temps.clear(owner=session)

    # -- what a client offers ------------------------------------------------------

    async def _offer(self, tools: tuple[OfferedTool, ...]) -> None:
        """Take what this client can run, or say one tool at a time why it was not taken.

        Attaching changes what every session in this kernel is shown, so nothing about it is silent:
        each refusal is an event with the rule it broke, and each session's tool set epoch moves once
        for the whole offer, whatever the number of tools (§10.8, E44/E46).

        A second ``Offer`` on one connection replaces the first — the client is the same client, and
        what it can run is whatever it last said. An offer that attaches nothing and finds nothing
        attached is the one case that leaves the table alone: there is no change to count.
        """
        if CLIENT_OFFER not in self.link.manifest.caps:
            self._say(OfferRefused(reason=Refusal.NO_CAPABILITY))
            return
        before = self._offers_now()
        client = before.attached.id if before.attached is not None else f"c{before.issued + 1}"
        attached, refused = specs_from(client, tools, taken={spec.name for spec in self._specs})
        for name, reason in refused:
            self._say(OfferRefused(name=name, reason=reason))
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
        """Take what this client now serves, and let the next round build the table around it.

        A new table and a new executor, made lazily on the round that reads it: a turn in flight keeps
        the table it was asked with, so "a call runs with the tools the round was asked with" is true for
        the rounds already running, without anything having to check (E175).
        """
        self._attached = attached
        self._resident = resident
        self._epoch += 1
        self._made = None
        self._made_key = None

    # -- what a round is given -----------------------------------------------------

    def table(self) -> tuple[ToolTable, ToolExecutor]:
        """The table this round is asked with: the program's tools, what a provider brought, and what a
        client offers.

        Rebuilt when either side moved — the process's set (a group materialised) or this connection's
        (an offer, a withdrawal) — and kept otherwise, because a table is what a round reads once and two
        rounds of one tool set must be the same bytes. The materialising handed to the executor is the
        process's own: a group a round asks for is brought up once for everybody (E175).
        """
        tools = self._tools()
        grew = 0 if tools is None else tools.version
        key = (grew, self._epoch)
        if self._made is None or self._made_key != key:
            base = self._specs if tools is None else tools.specs()
            available = () if tools is None else tools.available()
            catalog = ToolCatalog((*base, *self._attached), resident=(*RESIDENT, *self._resident))
            table = ToolTable(catalog=catalog, available=available)
            self._made = (table, ToolExecutor(table, materialise=None if tools is None else tools.materialise))
            self._made_key = key
        return self._made

    def _tools(self) -> Tools | None:
        """The process's tool table, or ``None`` for an assembly that has none — a hand-built one.

        Read from the assembly and not held, because it is the process's and not this tree's: two
        connections to one kernel ask one table for a group, and that is the point of it being a
        component of the process (E175).
        """
        found = self.assembly.instance(Declaration(Tools).key)
        return found if isinstance(found, Tools) else None

    # -- bringing back what a conversation asked for -------------------------------

    async def ready(self, session: str) -> None:
        """Wait until *session*'s groups are back, or until they have failed to come.

        Shielded, because the wait is a turn's convenience and the work is the tree's: a turn stopped
        while its tools are still coming up stops waiting, and the groups arrive for whatever asks next.
        """
        pending = self._recovering.get(session)
        if pending is not None:
            _ = await asyncio.gather(asyncio.shield(pending), return_exceptions=True)

    def _begin_recovery(self, session: str) -> None:
        """Start bringing back what this conversation asked for, without waiting for it.

        Started when the session is opened and not before the turn: ``Open`` has to answer now, and a
        session nobody talks to should not hold a connection open. What waits for it is the first round
        (:meth:`ready`), so the first request of a reopened conversation already has the tools it had
        (E176).
        """
        self._recovering[session] = asyncio.create_task(self._recover(session))

    def _forget_recovery(self, session: str) -> None:
        """Stop bringing back a conversation's groups: it closed, or this connection left.

        What it costs is one group arriving for nobody — the materialising is the process's and is not
        cancelled with the wait — which is cheaper than a task per closed conversation.
        """
        pending = self._recovering.pop(session, None)
        if pending is not None:
            _ = pending.cancel()

    async def _recover(self, session: str) -> None:
        """Bring back every group this conversation asked for, in parallel, and say what did not come.

        The range is two sets and their difference: what the conversation asked for, against what this
        program can still bring up. A group on the wrong side of that is a group the program no longer
        has, which is a failure worth saying — silence would leave a model calling a name that will never
        answer again.

        Neither answer here refuses ``Open`` or fails a turn: the groups that came are in the table, and
        the ones that did not are one notice for the model to read. A recovery that raised would be a
        task nobody awaits, so nothing in it is allowed out (E176).
        """
        try:
            await self._brought_back(session)
        except Exception:  # noqa: BLE001 — a background task that raised would print where nobody reads
            return

    async def _brought_back(self, session: str) -> None:
        """The work :meth:`_recover` does, in a body that may raise."""
        tools = self._tools()
        found = self._session(session)
        if tools is None or found is None:
            return
        available = {group for group, _summary in tools.available()}
        known = available | {group for group, _members in self.table()[0].catalog.groups()}
        failed: list[tuple[str, str]] = [(group, ABSENT) for group in sorted(found.disclosed - known)]
        wanted = sorted(found.disclosed & available)
        answers = await asyncio.gather(*(tools.materialise(group) for group in wanted), return_exceptions=True)
        brought = False
        for group, answer in zip(wanted, answers, strict=True):
            if isinstance(answer, BaseException):
                failed.append((group, UNAVAILABLE))
            elif not answer.specs:
                failed.append((group, EMPTY))
            else:
                brought = True
        if brought:
            # The block this conversation is asked with moved, and the fold is what the next process
            # reads its groups out of (E175, E177).
            _ = await found.notice_tools()
            _ = await found.host.trail.snapshot(session)
        if failed:
            text, summary = providers(failed)
            found.notices.say(PROVIDERS, text, summary=summary)

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
        boundary that makes a mid-turn route change legal (R7).
        """
        found = self._session(session)
        if found is None:
            self._say(RouteRefused(session=session, reason=Refusal.NO_SESSION))
            return
        known = found.host.routes
        if route and known.named(route) is None:
            self._say(RouteRefused(session=session, reason=Refusal.NO_ROUTE, route=route))
            return
        name, epoch = await found.set_route(route)
        asked = known.resolve(name)
        self._say(
            Routed(
                session=session,
                route="" if asked is None else asked.name,
                epoch=epoch,
            )
        )
        # A route that moved may not serve the model this conversation was asking for, so what it stands
        # at now is said again, resolved against the route it moved to.
        self._say_model(found)

    async def _persisting(self, session: str) -> None:
        """Write this conversation's route back into the file, or say why the file was left alone (E173).

        The kernel writes the file it was started from, wherever that file is: a client connected over a
        socket asks here and the write happens on the machine serving it, which is the only machine where
        the answer means anything. Every refusal is the writer's own — this method decides nothing about
        what a file may contain, and a route that cannot be resolved is one the file does not declare.
        """
        found = self._session(session)
        if found is None:
            self._say(SessionRefused(session=session, reason=Refusal.NO_SESSION))
            return
        route = found.host.routes.resolve(found.route)
        if found.host.writing is None or route is None:
            name = "" if route is None else route.name
            self._say(PersistRefused(session=session, route=name, problem=PersistProblem.NO_FILE))
            return
        written = found.host.writing.write_models(route.name, route.models)
        if written.problem is not None:
            self._say(PersistRefused(session=session, route=route.name, problem=written.problem))
            return
        self._say(Persisted(session=session, route=route.name, path=written.path))

    async def _model(self, session: str, model: str) -> None:
        """Point one session at a model of its route, or say why nothing moved.

        One refusal: a session this kernel does not serve. Everything else is the endpoint's to answer,
        because the file lists what a person wanted to name and a gateway may serve more (E170).
        """
        found = self._session(session)
        if found is None:
            self._say(SessionRefused(session=session, reason=Refusal.NO_SESSION))
            return
        _ = await found.set_model(model)
        self._say_model(found)

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
        unchanged — a parent pointer into nothing is one fact and it is already said once.

        ``replaces`` is bookkeeping, and it names two entries of *this* conversation and nothing else
        reads it — so a name that is not here would be a claim nothing could ever check
        again. It is refused the way the parent pointer is, before anything is written.
        """
        found = self._session(session)
        if found is None:
            self._say(CompactRefused(session=session, reason=Refusal.NO_SESSION))
            return
        if not text:
            self._say(CompactRefused(session=session, reason=Refusal.NOTHING_TO_KEEP))
            return
        if replaces is not None:
            missing = [named for named in replaces if await found.host.trail.entry(session, named) is None]
            if missing:
                self._say(CompactRefused(session=session, reason=Refusal.NOT_A_SPAN, entry=missing[0]))
                return
        try:
            entry = await found.host.trail.append(
                session, Summary(text=text, replaces=replaces), begin_after=begin_after
            )
        except ValueError:
            self._say(CompactRefused(session=session, reason=Refusal.NO_SUCH_ENTRY, entry=begin_after or ""))
            return
        _ = await found.note_history()
        self._say(Compacted(session=session, entry=entry.id, chars=len(text)))

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
            self._say(ExportRefused(session=session, reason=Refusal.NO_SESSION))
            return
        trail = found.host.trail
        end = begin_after
        if end is None:
            head = await trail.head(session)
            end = None if head is None else head.id
        try:
            messages = await trail.derive(session, end)
        except ValueError:
            self._say(ExportRefused(session=session, reason=Refusal.CHAIN_BROKEN, entry=end or ""))
            return
        if not messages:
            self._say(ExportRefused(session=session, reason=Refusal.NOTHING_TO_EXPORT))
            return
        transcript_text = transcript(messages)
        if len(transcript_text) > MAX_TEXT_CHARS:
            self._say(
                ExportRefused(session=session, reason=Refusal.TOO_BIG, chars=len(transcript_text), limit=MAX_TEXT_CHARS)
            )
            return
        kept = await found.host.documents.put(
            session, transcript_text, summary=f"conversation exported ({len(messages)} messages)"
        )
        self._say(DocumentStored(session=session, document=kept.id, chars=kept.chars))

    async def _prune(self, session: str, targets: tuple[str, ...], reason: str) -> None:
        """Write one entry saying these entries are no longer shown, or say why nothing was written.

        The prune goes where the conversation ends — its parent is the current head — because that is the
        only placement a projection can honour: a turn continues from the entry it is given and walks up
        from there, so a prune written anywhere else would hide nothing from the turn that follows it.
        That also makes the prune the new head, which is what a client's cursor moves to.

        Every target is checked before anything is written: a list naming one entry this session does not
        have writes nothing at all, because a prune that half happened is a history nobody can account
        for.

        What the model is told about it is a line the session *owes*: the command may arrive while a turn
        is in flight, and a message appended inside a round would sit between an answer and its results.
        So the elision is one sentence on the next boundary, and a
        history that stops showing something stops quietly no more (E149).
        """
        found = self._session(session)
        if found is None:
            self._say(PruneRefused(session=session, reason=Refusal.NO_SESSION))
            return
        if not targets:
            self._say(PruneRefused(session=session, reason=Refusal.NOTHING_TO_PRUNE))
            return
        trail = found.host.trail
        for target in targets:
            if await trail.entry(session, target) is None:
                self._say(PruneRefused(session=session, reason=Refusal.NO_SUCH_TARGET, entry=target))
                return
        head = await trail.head(session)
        entry = await trail.append(
            session,
            Withheld(targets=targets, reason=reason, by=BY_PERSON),
            begin_after=None if head is None else head.id,
        )
        # What the model is told is owed and written at the next boundary: a prune may land while a turn
        # is in flight, and a message appended inside a round would sit between an answer and its results.
        found.notices.say(
            ELIDED,
            elided(len(targets), by=BY_PERSON, reason=reason),
            summary=elision_summary(len(targets), by=BY_PERSON),
        )
        # The conversation under the prompt moved, so the number a provider cache keys on moves with it:
        # a prune is one of the things that makes a prefix miss, and a miss nobody counted is a miss
        # nobody can account for (§10.5, E149).
        _ = await found.note_history()
        self._say(Pruned(session=session, entry=entry.id, targets=targets, reason=reason))

    # -- turns ---------------------------------------------------------------------

    async def _turning(self, command: Submit | Retry) -> None:
        """The two commands that start a turn: a message with text, and a retry with none.

        Both go through :meth:`_submit`, so "one turn at a time in a session" and the kernel's own
        ceiling answer a retry exactly as they answer a message (E158).
        """
        match command:
            case Submit(session=session, text=text, kind=kind, begin_after=begin_after):
                await self._submit(session, text, kind, begin_after)
            case Retry(session=session, begin_after=begin_after):
                await self._retry(session, begin_after)

    async def _retry(self, session: str, begin_after: str | None) -> None:
        """Ask the model again from the chain's end, and not from a place the chain does not have.

        A retry that names no place continues where the conversation stands, which is what "again" means;
        a conversation with nothing in it has nothing to ask again, and is told so, with no request
        carrying an empty history (E158).
        """
        found = self._session(session)
        if found is None:
            self._say(TurnRefused(session=session, reason=Refusal.NO_SESSION, holder=0))
            return
        if begin_after is None:
            head = await found.host.trail.head(session)
            if head is None:
                self._say(TurnRefused(session=session, reason=Refusal.NOTHING_TO_RETRY, holder=0))
                return
            begin_after = head.id
        await self._start(found, session, "", TurnKind.RETRY, begin_after)

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
        "waiting on turn 7", with no message invented.

        A turn is also refused when the kernel is already running as many as it may, which is a different
        answer from ``busy``: that one is about *this* conversation, and the ceiling is about the process.
        Nothing is queued for either — a turn that cannot start is said so at once (E156).
        """
        found = self._session(session)
        if found is None:
            self._say(TurnRefused(session=session, reason=Refusal.NO_SESSION, holder=0))
            return
        if kind in _KERNELS_OWN:
            # Two kinds are this kernel's to start and never a client's to name: a goal turn is the plan
            # saying the work is not done, and a retry is a person saying "ask again" through the command
            # of that name (E146/E158). An aside is a client's to ask for — it is a message that writes
            # nothing the conversation keeps — so it arrives here as a ``Submit`` (E163).
            self._say(TurnRefused(session=session, reason=_KERNELS_OWN[kind], holder=0))
            return
        await self._start(found, session, text, kind, begin_after)

    async def _start(self, found: Session, session: str, text: str, kind: TurnKind, begin_after: str | None) -> None:
        """What every turn takes, whoever asked it: a model to ask, the session's one slot, a place in the
        kernel's budget — and then a task of its own.

        It is separate from :meth:`_submit` because a retry is this kernel's own kind: what a client may
        not name, the kernel may start, and the checks afterwards are the same ones either way (E158).
        """
        if not found.host.model.installed:
            self._say(TurnRefused(session=session, reason=Refusal.NO_MODEL, holder=0))
            return
        route = found.host.routes.resolve(found.route)
        if route is not None and route.asking(found.model) is None:
            # Nothing to ask: this conversation never chose a model of its route, or the file no longer
            # lists the one it chose. Refused before a turn begins, so no prompt is built for a request
            # that has no id to carry (E170). A tree with no route at all is a different absence, and it
            # is answered where the request is built.
            self._say(TurnRefused(session=session, reason=Refusal.NO_MODEL, holder=0))
            return
        live = found.turns()
        if live:
            self._say(TurnRefused(session=session, reason=Refusal.BUSY, holder=live[0].turn_id))
            return
        turned = await self._begun(found, session, kind, begin_after)
        if turned is None:
            return
        task = asyncio.create_task(self._turn(turned, text))
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)

    async def _begun(self, found: Session, session: str, kind: TurnKind, begin_after: str | None) -> TurnCtx | None:
        """Start the turn, or say why there is none: an entry that is not there, or a kernel at its ceiling.

        The two refusals are different facts and are said as such: ``no-such-entry`` is about the argument
        a client sent, and ``too-many-turns`` is about this kernel's budget — a client that switches on
        the reason can retry one and must not retry the other (E156).
        """
        try:
            return await found.begin_turn(asker=self._asker, begin_after=begin_after, kind=kind)
        except ValueError:
            self._say(TurnRefused(session=session, reason=Refusal.NO_SUCH_ENTRY, entry=begin_after or ""))
            return None
        except TooManyTurns as refused:
            self._say(TurnRefused(session=session, reason=Refusal.TOO_MANY_TURNS, limit=refused.limit))
            return None

    async def _turn(self, turned: TurnCtx, text: str) -> None:
        """Run one turn to its end, and then whatever the goal switch says about the next one.

        A turn that raised has already announced that it ended badly and has already stopped being the
        session's live turn — the runner's ``finally`` does both — so there is nothing left to close
        here. Letting the exception out of the task instead would only hand it to a task nobody awaits,
        where the loop's default handler would print it onto the terminal the kernel is not allowed to
        write to.

        The chain is here, because this is where a turn is *started*: a goal turn
        is one more turn, with its own address, its own cost and its own end row (§10.4, E146).
        """
        while True:
            runner = TurnRunner(turned, giving=self)
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
        if not session.goal or turned.kind in (TurnKind.ASIDE, TurnKind.COMPACT):
            return None
        if end.reason or not end.acted or not session.todos.reminder() or session.turns():
            return None
        try:
            return await session.begin_turn(asker=turned.asker, begin_after=end.ends_at, kind=TurnKind.GOAL)
        except ValueError, TooManyTurns:
            # A plan that cannot continue is not a failure: an entry that is gone and a kernel already at
            # its ceiling both mean this turn is the last one for now (E156).
            return None

    # -- the addresses -------------------------------------------------------------

    def _declaration(self, session: str) -> Declaration:
        """How a session with this id is declared: under the host, with the id as its tail."""
        return Declaration(Session, instance=f"{self.host}/{session}", kwargs={"session_id": session})

    def _host(self) -> Host | None:
        """The host this kernel serves under, or ``None`` when the assembly has it down.

        What a read of another conversation needs and a served session does not: seeding writes rows for
        a session this kernel has not admitted, so the journal is reached through the host, since there
        is no session to point at it.
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
        """Every session this *tree* serves, in the order they were admitted.

        The same walk a listing does, narrowed to this host: sessions are declared as ``ws/name``, so the
        qualifier is what tells one tree's conversation from another's. A bare identity match would find
        every tree's, and an offer made to one tree would tell the others their tool set moved.
        """
        identity = Declaration(Session).identity
        declared = self.assembly.keys()
        prefix = f"{self.host}/"
        found = [
            self.assembly.instance(key)
            for key in declared
            if key.identity == identity and key.instance.startswith(prefix)
        ]
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
        listed: list[tuple[str, str, str]] = []
        for name in held:
            state = await host.trail.state_of(name)
            listed.append((name, await self._head(host.trail, name), state.get("name", "")))
        gone = set(held)
        for session in self._sessions():
            if session.session_id in gone:
                continue
            # An open conversation the journal does not list is either one nothing was said in yet — which
            # is "I am here" and belongs in the listing — or one somebody deleted, which does not.
            rows = await host.trail.entries(session.session_id)
            if any(kind_of(row.payload) == DELETED_TAG for row in rows):
                continue
            listed.append((session.session_id, "", session.name))
        return tuple(ListedSession(session=name, head=head, name=called) for name, head, called in listed)

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
