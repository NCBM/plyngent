"""One answer's tool calls, from the text a model wrote to the results three readers share.

This is the only place that knows both halves — the catalog's declarations and a handler's expectations
— and it exists so that neither has to know about the other. Four things happen to a call, in this order,
and the order is the design:

* **decode.** The model's text becomes the tool's argument package, and text that is not one is an
  argument error *of that tool*: the call never ran, so nothing is owed and the fix is a different call.
* **approve.** The posture a tool's declarations ask for (:meth:`ToolSpec.approval`) is put to a person
  through the host's approval table: once per what the tool is remembered by, or every time when it says
  so. The table is the host's, so an approval is not asked again by the next session — and one scoped to
  a session names its session in the key rather than living in a second table.
* **run, under the call's own address.** The handler is handed an
  :class:`~plyngent.kernel.env.Env` built from the resources the tool declared, so it reaches exactly
  what its declaration says (§10.1). Every await inside it goes through the turn's cancellation token
  through that address, so a turn that is stopped stops its tool too — and the tool's result still comes
  back, because a cancelled call that ran is a result the history is owed (``LIMIT``, never retried:
  §10.11).
* **say so.** :class:`~plyngent.wire.event.ToolCalled` and
  :class:`~plyngent.wire.event.ToolReturned` are emitted here, around the run. This is the only layer
  that knows both what the model meant by the call and what came of it, and the one field a person
  needs while a call is running — the model's own reason — is inside the arguments, which only this
  layer has decoded.

**One answer is a batch, and a batch is a small graph** (§10.12). Every call is decoded before any of them
runs, because the paths a call will touch are read out of its arguments — and those paths are what decides
which calls may overlap: a write conflicts with every call on that path, a read only with a write on it,
the direction is the order the model wrote them in, and a call that names no path at all waits for this
answer's file work before it starts. The graph says what *may* run at once; the host's locks
(:mod:`plyngent.kernel.locks`) are what actually hold, so two turns of two conversations obey the same
rule. The results are collected in the model's order rather than the order they came back in: one answer
has to produce one history, whoever finished first.

Three tools ask through a policy instead of through this gate, because their question is about run state
this layer cannot see: whether a program is on the denylist (a command policy that reads the argv),
whether a terminal is still alive (a PTY policy that reads the session). They declare ``Posture.AUTO``
where the baseline would otherwise ask, so that one call never becomes two questions. The day those
policies raise a posture rather than asking, the override goes.

What is *not* here: a cap on the result. ``meta.chars`` already measures what the model will read, and
truncation is "move the body to the document store and keep the handle" — which needs the store, so it
is not a number this layer invents. The paths inside an ``execute`` are here, though: a tool whose
arguments are a tree declares what it finds in it (:attr:`~plyngent.kernel.toolspec.ToolSpec.holds`), so
the graph sees the file a node reads and the file a redirect writes, and two calls that touch one of
those are ordered like any other two. What it still cannot see is a path a program chooses while it
runs: a call that spells none and lets its argv decide is invisible to the schedule, and two of those
that depend on each other are two answers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.client import declared
from plyngent.kernel.env import Env
from plyngent.kernel.failure import Failure
from plyngent.kernel.history import History, Unreadable
from plyngent.kernel.listing import walk
from plyngent.kernel.paths import display, key
from plyngent.kernel.toolresult import Document, Effects, Hint, ToolResult
from plyngent.kernel.toolspec import ArgsError, Mutation, Posture, Remember, Res, Role, argument
from plyngent.kernel.workspace import PathRefused
from plyngent.wire.ask import By, Confirmation, as_agreed
from plyngent.wire.event import Changed, DocumentStored, ToolCalled, ToolReturned

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.catalog import ToolCatalog, ToolView
    from plyngent.kernel.model import Call
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.toolresult import Part
    from plyngent.kernel.toolspec import ToolSpec
    from plyngent.wire.event import AnyEvent

#: Which access a path argument is resolved with, when it is resolved for an approval's key.
_ACCESS: dict[Role, AccessMode] = {Role.READ: AccessMode.READ, Role.WRITE: AccessMode.WRITE}

#: How many calls of one answer may run at the same time. A cap rather than a promise: the graph decides
#: what may overlap, and this decides how much of it is worth doing at once — a batch of forty reads is
#: forty open files, which is a number to bound rather than to discover.
CONCURRENCY = 8

#: The one tool name this layer may write into a hint: showing a group is what a refusal for a tool
#: nobody has been shown points at, and that tool exists to be named by exactly such a line (§10.4).
ASKING = "tool_require"

#: How much of one difference an event carries. Past this the patch is cut and the event says so; the
#: whole of it is still in the store under the version the event names, so nothing is lost — a limit on
#: what crosses one wire, not on what is kept.
PATCH_CHARS = 20_000

#: How many files one sweep may compare before it gives up on taking a version. A program's own paths are
#: invisible (§10.12), so the tree itself is the only witness — and hashing a checkout the size of a
#: kernel on every command would be a stall nobody asked for. Past this the call keeps its record and no
#: version, which is the same degradation as a host with no store.
SWEEP_PATHS = 5_000


@dataclass(frozen=True, slots=True)
class _Swept:
    """What one sweep came to: the paths it found different, or how big the tree was when it gave up.

    One value rather than two answers, because the two are the same question asked twice: a sweep either
    compared the tree and knows what moved, or it looked at a tree too big to compare and knows how big.
    """

    changed: tuple[Path, ...] = ()
    #: How many files the tree held when it was over the bound, and ``0`` when the tree was compared.
    counted: int = 0


@dataclass(frozen=True, slots=True)
class Held:
    """One path a call declared, resolved, and the role the declaration gave it.

    The role is kept rather than dropped: a batch orders the calls of one answer by what they do to a
    path, and a lock that forgets whether a call reads or writes has nothing to order by (§10.12).
    """

    path: Path
    role: Role


@dataclass(slots=True)
class _Call:
    """One call of a batch, as the schedule sees it: what it touches, what it waits for, what came out."""

    call: str
    name: str
    spec: ToolSpec | None
    args: object
    #: The arguments as the model spelled them. Kept because a call that leaves this process has to
    #: declare what it sent, and the decoded package is not that text.
    raw: str = ""
    #: The answer when nothing runs — a name nobody has, arguments that are not this tool's.
    refusal: ToolResult | None = None
    held: tuple[Held, ...] = ()
    #: Whether the workspace refused a path this call named, which its own handler is about to report.
    refused: bool = False
    #: The calls that have to finish before this one may start.
    after: set[int] = field(default_factory=set)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    result: ToolResult | None = None

    @property
    def runs(self) -> bool:
        """Whether this call has a handler to reach: one that answers without running does not."""
        return self.refusal is None

    @property
    def touches_files(self) -> bool:
        """Whether this call changes or reads the workspace, as far as its own declaration says."""
        return bool(self.held) or self.refused


class ToolExecutor:
    """Runs one call at a time, against one catalog.

    A batch — several calls from one answer, ordered by the paths they touch — is the scheduler's
    business and not this object's: one call is the unit here, and it is the unit every caller has.
    """

    _catalog: ToolCatalog

    def __init__(self, catalog: ToolCatalog) -> None:
        self._catalog = catalog

    async def batch(self, ctx: TurnCtx, calls: Sequence[tuple[str, Call]]) -> tuple[ToolResult, ...]:
        """Run one answer's calls as far in parallel as the paths they hold allow.

        ``calls`` are ``(address, call)`` pairs in the order the model asked for them, and the answers come
        back in that order whatever order they finished in: an answer's results are part of the history,
        and two runs of one script have to write the same history. ``address`` is this call's segment of
        the turn's scope (``tool-1``) — the turn knows how many calls it has made — and the same segment is
        what a frontend pairs :class:`~plyngent.wire.event.ToolCalled` with
        :class:`~plyngent.wire.event.ToolReturned` by.

        What a call may reach is the session's view, built once for the batch: the tools the round was
        asked with. A call naming a tool it cannot see is answered rather than run, and the answer says
        which group that tool is in — naming it here is this layer's job, because it is the only one
        that holds both the registry and the session.
        """
        view = self._catalog.view(ctx.session.disclosed)
        nodes = _prepared(ctx, self._catalog, view, calls)
        _ordered(nodes)
        await self._run(ctx, nodes, view)
        answered: list[ToolResult] = []
        for node in nodes:
            result = node.result
            assert result is not None, "a batch answers every call it was given"
            answered.append(result)
        return tuple(answered)

    async def _run(self, ctx: TurnCtx, nodes: list[_Call], view: ToolView) -> None:
        """Start each call once what it waits for has finished, and no more than ``CONCURRENCY`` at once."""
        breathing = asyncio.Semaphore(CONCURRENCY)

        async def start(index: int) -> None:
            node = nodes[index]
            try:
                for waited in sorted(node.after):
                    _ = await nodes[waited].finished.wait()
                async with breathing:
                    node.result = await self._answer(ctx, node, view)
            finally:
                # Set even by a cancellation: a call that stopped is still finished as far as the calls
                # waiting behind it are concerned, and leaving them waiting would hang the batch.
                node.finished.set()

        _ = await asyncio.gather(*(start(index) for index in range(len(nodes))))

    # -- one call --------------------------------------------------------------------

    async def _answer(self, ctx: TurnCtx, node: _Call, view: ToolView) -> ToolResult:
        """Say what this call is about to do, then run it — or answer what it is instead."""
        session, turn = ctx.session.session_id, ctx.turn_id
        self._emit(
            ctx,
            ToolCalled(session=session, turn=turn, call=node.call, tool=node.name, intention=_why(node.args)),
        )
        result = node.refusal if node.refusal is not None else await self._ran(ctx, node, view)
        result = _reachable(result, self._catalog, view)
        self._emit(
            ctx,
            ToolReturned(
                session=session,
                turn=turn,
                call=node.call,
                tool=node.name,
                status=result.status.value,
                summary=result.summary,
                elapsed_ms=result.meta.elapsed_ms,
            ),
        )
        # A body that did not fit was written to the store by the tool that produced it, before this
        # result existed, so a handle that is here names content that is already there. Saying it out
        # loud is for the person watching: what they are not seeing still exists, under this name.
        for part in result.body:
            if isinstance(part, Document):
                self._emit(ctx, DocumentStored(session=session, document=part.document_id, chars=part.chars))
        await self._shown(ctx, node, result)
        return result

    async def _shown(self, ctx: TurnCtx, node: _Call, result: ToolResult) -> None:
        """Tell whoever is watching what this call left different, and tell the model nothing.

        The model's account of a call is its result, in the tool's own words; a second account would be a
        second thing to read and to keep in step. This is the other reader's: the difference itself, out
        of the store rather than off the disk, so the answer cannot drift from what the call did.

        A store that cannot answer leaves this unsaid rather than failing the call: the change happened,
        and a viewer that is missing one difference is a viewer, not a wrong result.
        """
        store, revision = ctx.host.history, result.effects.revision
        if store is None or not revision:
            return
        try:
            parent = await store.parent(revision)
            patch = await store.diff(parent, revision)
            paths = await store.changed(revision)
        except Unreadable:
            return
        self._emit(
            ctx,
            Changed(
                session=ctx.session.session_id,
                turn=ctx.turn_id,
                call=node.call,
                tool=node.name,
                revision=revision,
                paths=paths,
                patch=patch[:PATCH_CHARS],
                truncated=len(patch) > PATCH_CHARS,
            ),
        )

    async def _ran(self, ctx: TurnCtx, node: _Call, view: ToolView) -> ToolResult:
        """Ask about this call if its declarations say to, then run it — or answer the refusal."""
        spec, args = node.spec, node.args
        assert spec is not None, "a call with nothing to run is answered without reaching here"
        turned_down = await self._approve(ctx, node)
        if turned_down is not None:
            return turned_down
        reads, writes = _locks(node.held)
        env = Env(ctx, spec.uses, node.call, tool=spec.name, tools=view)
        try:
            # The lock is the host's, so this is where two turns in two conversations agree about one file
            # — the graph above only orders the calls of a single answer.
            async with ctx.host.locks.hold(reads=reads, writes=writes):
                result = await env.scope().guard(spec.run(env, args))
                # Inside the lock on purpose: a version has to be taken while the paths still hold what
                # this call left, and a lock released first is a file another call may have written.
                return await self._versioned(ctx, node, result, scope=env.scope())
        except Cancelled:
            # A call that was stopped still left this process, so it still declares what it sent. The
            # handler normally records that, and a stopped call never gets as far as a result of its own.
            return ToolResult.of(Failure.LIMIT, "stopped", effects=_sent(spec, node.raw))

    async def _versioned(self, ctx: TurnCtx, node: _Call, result: ToolResult, *, scope: Scope) -> ToolResult:
        """Keep what this call changed as one version, and answer the result that names it.

        A store that is not there, a call that changed no path, and a store that could not be written
        all leave the effect as it was — with no version — because the change happened whether or not
        anything managed to keep it, and a result that failed over a bookkeeping problem would be
        reporting the wrong thing about the call.

        The paths come from the handler's own ``Effects``, which is the list this kernel trusts for
        "what is no longer what you last saw". A tool that cannot say what it changed — a program was
        started, and its argv names no files (§10.12) — is swept instead: the tree is enumerated by the
        kernel's own rules and compared with the version in force, so what a build or a formatter wrote
        is still kept, under the same one-version-per-call rule.

        A sweep that gave up says so in the effect (:attr:`~plyngent.kernel.toolresult.Effects.unswept`),
        because "there is no version" and "this call was too big to compare" are not the same fact and a
        person reading the log later is the one who has to know which of the two happened.
        """
        store = ctx.host.history
        if store is None:
            return result
        changed = result.effects.changed
        try:
            if not changed and node.spec is not None and node.spec.mutates & Mutation.WORKSPACE:
                swept = await self._swept(ctx, store, scope=scope)
                if swept.counted:
                    return replace(result, effects=replace(result.effects, unswept=swept.counted))
                changed = swept.changed
            if not changed:
                return result
            revision = await store.record(changed, says=f"{node.name} {node.call}", scope=scope)
        except Unreadable:
            return result
        return replace(result, effects=replace(result.effects, revision=revision))

    async def _swept(self, ctx: TurnCtx, store: History, *, scope: Scope) -> _Swept:
        """What a program left behind, read off the tree because the call cannot say.

        The walk is the kernel's own (the same one a listing uses: its ignore rules, its hidden
        directories, its denylist), never ``git add`` — a sweep by git would apply a second set of
        ignore rules to the same tree and record files this kernel deliberately does not look at.

        A tree too big to compare is not compared: the effect still says the program ran, and the call
        costs no stall. That bound is the honest one — a version nobody could afford to take is worse
        than a record that says where it stopped — and the number it gives up at is the answer rather
        than silence, because a log that says nothing about a version reads like a call that changed
        nothing.
        """
        root = ctx.host.workspace.root
        listing = await scope.blocking(walk, ctx.host.workspace, root)
        files = [root / entry.relative for entry in listing.entries if not entry.is_dir]
        if len(files) > SWEEP_PATHS:
            return _Swept(counted=len(files))
        return _Swept(changed=await store.differing(await store.head(scope=scope), files, scope=scope))

    async def _approve(self, ctx: TurnCtx, node: _Call) -> ToolResult | None:
        """Put this call to a person when its posture says to; ``None`` when nothing has to be asked.

        A call naming a path the workspace will not resolve is let through unasked and refused by its own
        handler: asking somebody to allow a path that is refused before it is looked at would be the wrong
        question.
        """
        spec, args = node.spec, node.args
        assert spec is not None
        asking = spec.asking(args)
        if asking.posture is Posture.AUTO or node.refused:
            return None
        subjects, scoped = _subjects(ctx, spec, node.args, asking.remember, node.held)
        approvals = ctx.host.approvals
        if not approvals.needs(asking.posture, remember=asking.remember, key=scoped, severity=asking.severity):
            return None
        question = f"allow {spec.name}" + (f" on {subjects}" if subjects else "") + "?"
        answer = as_agreed(
            await ctx.ask(
                Confirmation(prompt=question, severity=asking.severity),
                tool=spec.name,
                call=node.call,
                at="approve",
            )
        )
        if not answer.accepted:
            return ToolResult.unagreed(answer, about="this call")
        if answer.by is not By.POLICY:
            # A policy's yes is not a person's: it is what this program was started to do, so it says
            # nothing about the next call and is not written into what this host has been told.
            approvals.agree(asking.posture, remember=asking.remember, key=scoped, severity=asking.severity)
        return None

    def _emit(self, ctx: TurnCtx, event: AnyEvent) -> None:
        """Say what a call is doing, through the same channel every other claim travels on."""
        ctx.host.human.emit(event)


def _sent(spec: ToolSpec, arguments: str) -> Effects:
    """What this call sent out of this process, or nothing when it stayed here.

    A tool that declares :attr:`~plyngent.kernel.toolspec.Res.CLIENT` has its effect in somebody
    else's process, so the only honest record is the declaration that it was asked for — which is why
    it is written even for a call that was stopped before it could come back (§10.8).
    """
    if not spec.uses & Res.CLIENT:
        return Effects()
    return Effects(external=(declared(spec, arguments),))


def _read(spec: ToolSpec, arguments: str) -> tuple[object, None] | tuple[None, ToolResult]:
    """The call's arguments, or what to answer without running anything."""
    try:
        return spec.decode(arguments), None
    except ArgsError as exc:
        return None, ToolResult.of(Failure.ARGS, str(exc))


def _why(args: object) -> str:
    """The model's own account of why it is calling, or nothing when the tool does not ask for one."""
    return str(argument(args, "intention") or "").strip()


def _subjects(
    ctx: TurnCtx, spec: ToolSpec, args: object, remember: Remember | None, held: tuple[Held, ...]
) -> tuple[str, tuple[str, ...]]:
    """What an approval of this call is about: something to say, and something to remember it by.

    A path is keyed by :func:`~plyngent.kernel.paths.key` — the same normalization a lock uses — so
    ``a/../b`` and ``b`` are one approval. A session names itself, because the table belongs to the host
    and an approval in one conversation must not answer for another. A host-scoped approval covers
    everything, so its key is empty: there is only one host.

    A tool whose subject is one of its arguments (:attr:`~plyngent.kernel.toolspec.ToolSpec.subject`)
    names that value instead, so the person is asked about the one skill or directory the call is about
    rather than about the tool in general.

    A tool that says where its own paths are (:attr:`~plyngent.kernel.toolspec.ToolSpec.holds`) is keyed
    by them like any other: the question a batch's locks answer and the question a person answers have to
    be about the same paths, or a yes would be remembered for something the call never touched.
    """
    if spec.holds is not None or any(role in _ACCESS for role in (spec.roles or {}).values()):
        if not held:
            return "", ()
        names = ", ".join(display(item.path, root=ctx.host.workspace.root) for item in held)
        return names, tuple(sorted(key(item.path) for item in held))
    if spec.subject:
        named = argument(args, spec.subject)
        if isinstance(named, str) and named:
            return named, (named,)
    if remember is Remember.SESSION:
        return ctx.session.session_id, (ctx.session.session_id,)
    return "", ()


def _paths(ctx: TurnCtx, spec: ToolSpec, args: object) -> tuple[tuple[Held, ...], bool]:
    """Every path this call declares, resolved with its role, and whether any of them was refused.

    A call's paths come from one of two places and are resolved the same way either way: a table of
    argument names (:attr:`~plyngent.kernel.toolspec.ToolSpec.roles`) for a tool whose paths are its own
    fields, and a walk over its arguments (:attr:`~plyngent.kernel.toolspec.ToolSpec.holds`) for one whose
    arguments are a tree. Resolution is per path rather than all-or-nothing: a call naming two paths and
    getting one refused still reads or writes the other, so what it holds is the paths it will really
    touch — under the role it declared for each, because that is what a batch orders calls by. An argument
    that is empty names no path at all: a tool with two locators leaves the other one blank, and a blank is
    not a request for the workspace root. A refused path is reported rather than guessed at: nothing
    touches a path the workspace will not resolve, so there is no lock to take for it, but "this call named
    a path and it was refused" is a different fact from "this call named no path", and both the approval
    and the batch read the difference.
    """
    workspace = ctx.host.workspace
    grants = ctx.session.grants
    held: list[Held] = []
    refused = False
    named = [(argument(args, name), role) for name, role in (spec.roles or {}).items()]
    if spec.holds is not None:
        named.extend((reach.value, reach.role) for reach in spec.holds(args))
    for value, role in named:
        if role not in _ACCESS or not isinstance(value, str) or not value:
            continue
        try:
            held.append(Held(workspace.resolve(value, _ACCESS[role], grants=grants), role))
        except PathRefused:
            refused = True
    return tuple(held), refused


def _prepared(ctx: TurnCtx, catalog: ToolCatalog, view: ToolView, calls: Sequence[tuple[str, Call]]) -> list[_Call]:
    """Read every call of one answer before any of them runs.

    A batch has to be decoded as a whole, because what a call will touch is inside its arguments and the
    schedule is made of that. Answers come out of it here and not later — a name this session cannot see,
    a name nobody has, and arguments that are not this tool's are all answered without running, and a
    call that answers without running touches nothing, so the schedule leaves it out.

    The two ways a name can be unknown are answered differently on purpose: a tool this kernel has but
    this session was never shown says which group it is in, because that is the one thing the caller can
    do something about; a tool nobody has says so and nothing more.
    """
    nodes: list[_Call] = []
    for address, call in calls:
        spec = view.get(call.tool)
        if spec is None:
            unanswered = _unreachable(catalog, call.tool)
            nodes.append(_Call(call=address, name=call.tool, spec=None, args=None, refusal=unanswered))
            continue
        args, refusal = _read(spec, call.arguments)
        node = _Call(call=address, name=call.tool, spec=spec, args=args, refusal=refusal, raw=call.arguments)
        if args is not None:
            node.held, node.refused = _paths(ctx, spec, args)
        nodes.append(node)
    return nodes


def _reachable(result: ToolResult, catalog: ToolCatalog, view: ToolView) -> ToolResult:
    """A result whose hints point at tools this session can actually call.

    A hint names the call that would have worked, and a hint naming a tool nobody was shown is a
    suggestion the model cannot take: it would be told to use ``listdir`` by a kernel that did not offer
    it. So the hint is rewritten to the one call that does work — the group, and the tool that shows it
    — and it is rewritten here rather than in each handler because no handler knows what the session was
    shown, and forty handlers deciding it again is forty answers (§10.3).

    Nothing else about a hint changes: the reason survives, because the reason is what makes the hint
    worth following, and the argument patch does not, because it belonged to a call nobody can make.
    """
    if not any(isinstance(part, Hint) and part.tool not in view for part in result.body):
        return result
    body = tuple(_reachable_part(part, catalog, view) for part in result.body)
    return replace(result, body=body, meta=replace(result.meta, chars=0))


def _reachable_part(part: Part, catalog: ToolCatalog, view: ToolView) -> Part:
    """One part of a body, with a hint about an unreachable tool pointing at the group instead."""
    if not isinstance(part, Hint) or part.tool in view:
        return part
    group = catalog.group_of(part.tool)
    if group is None:
        return part
    return Hint(tool=ASKING, given=(("group", group),), why=part.why)


def _unreachable(catalog: ToolCatalog, name: str) -> ToolResult:
    """What to answer a call naming a tool this session cannot reach.

    A tool that exists in another group is the interesting case: the model was told there are groups it
    has not been shown, so the answer names the group and the one call that shows it. Naming
    :data:`ASKING` here is this layer's privilege and not a leak — the tool it names is the one whose
    whole purpose is to be named by the kernel's own refusals (§10.3).
    """
    group = catalog.group_of(name)
    if group is None:
        return ToolResult.of(Failure.ARGS, f"no tool is named {name!r}")
    return ToolResult.of(
        Failure.ARGS,
        f"{name!r} is not available yet: it is in the {group!r} group",
        hints=(Hint(tool=ASKING, given=(("group", group),), why="show that group first"),),
    )


def _ordered(nodes: list[_Call]) -> None:
    """Give every call of a batch the calls it has to wait for (§10.12).

    Three rules, and the third is the one that is easy to get wrong: a call that names no path waits for
    *every* file call of the answer, not only the ones written before it. A command's own paths are
    invisible here — ``execute`` keeps them inside a tree — so the only honest thing to promise it is the
    file state the whole answer produces, rather than a snapshot of an answer half applied.
    """
    file_work = {index for index, node in enumerate(nodes) if node.runs and node.touches_files}
    for index, node in enumerate(nodes):
        if not node.runs:
            continue
        if not node.touches_files:
            node.after = set(file_work)
            continue
        node.after = {earlier for earlier in range(index) if earlier in file_work and _conflict(nodes[earlier], node)}


def _conflict(earlier: _Call, later: _Call) -> bool:
    """Whether these two calls cannot run at the same time, given that ``earlier`` was written first.

    A write conflicts with every call on its path; a read only with a write on it. The direction is the
    model's own order, which is what makes "write then read one file" read the write and "read then
    write" read what was there before it.
    """
    return bool(_written(earlier.held) & _touched(later.held) or _written(later.held) & _touched(earlier.held))


def _touched(held: tuple[Held, ...]) -> set[str]:
    """Every path key these paths name, read or written."""
    return {key(item.path) for item in held}


def _written(held: tuple[Held, ...]) -> set[str]:
    """The path keys these paths name as writes."""
    return {key(item.path) for item in held if item.role is Role.WRITE}


def _locks(held: tuple[Held, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The lock keys this call reads and the ones it writes; a path it writes is never also read."""
    writes = _written(held)
    reads = {key(item.path) for item in held if item.role is Role.READ}.difference(writes)
    return tuple(sorted(reads)), tuple(sorted(writes))
