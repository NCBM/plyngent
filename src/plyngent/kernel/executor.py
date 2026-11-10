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
is not a number this layer invents. Nor are the paths inside an ``execute``: its ``argv``, its pipes and
its ``stdin`` live in a tree of steps, and ``roles`` is flat, so the graph sees a command's ``cwd`` and
nothing else. Two calls that change files and depend on each other through a program are two answers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.env import Env
from plyngent.kernel.failure import Failure
from plyngent.kernel.paths import display, key
from plyngent.kernel.toolresult import Document, ToolResult
from plyngent.kernel.toolspec import ArgsError, Posture, Remember, Role, argument
from plyngent.kernel.workspace import PathRefused
from plyngent.wire.event import DocumentStored, ToolCalled, ToolReturned

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from plyngent.kernel.catalog import ToolCatalog
    from plyngent.kernel.model import Call
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.toolspec import ToolSpec
    from plyngent.wire.event import AnyEvent

#: Which access a path argument is resolved with, when it is resolved for an approval's key.
_ACCESS: dict[Role, AccessMode] = {Role.READ: AccessMode.READ, Role.WRITE: AccessMode.WRITE}

#: How many calls of one answer may run at the same time. A cap rather than a promise: the graph decides
#: what may overlap, and this decides how much of it is worth doing at once — a batch of forty reads is
#: forty open files, which is a number to bound rather than to discover.
CONCURRENCY = 8


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
        """
        nodes = _prepared(ctx, self._catalog, calls)
        _ordered(nodes)
        await self._run(ctx, nodes)
        answered: list[ToolResult] = []
        for node in nodes:
            result = node.result
            assert result is not None, "a batch answers every call it was given"
            answered.append(result)
        return tuple(answered)

    async def _run(self, ctx: TurnCtx, nodes: list[_Call]) -> None:
        """Start each call once what it waits for has finished, and no more than ``CONCURRENCY`` at once."""
        breathing = asyncio.Semaphore(CONCURRENCY)

        async def start(index: int) -> None:
            node = nodes[index]
            try:
                for waited in sorted(node.after):
                    _ = await nodes[waited].finished.wait()
                async with breathing:
                    node.result = await self._answer(ctx, node)
            finally:
                # Set even by a cancellation: a call that stopped is still finished as far as the calls
                # waiting behind it are concerned, and leaving them waiting would hang the batch.
                node.finished.set()

        _ = await asyncio.gather(*(start(index) for index in range(len(nodes))))

    # -- one call --------------------------------------------------------------------

    async def _answer(self, ctx: TurnCtx, node: _Call) -> ToolResult:
        """Say what this call is about to do, then run it — or answer what it is instead."""
        session, turn = ctx.session.session_id, ctx.turn_id
        self._emit(
            ctx,
            ToolCalled(session=session, turn=turn, call=node.call, tool=node.name, intention=_why(node.args)),
        )
        result = node.refusal if node.refusal is not None else await self._ran(ctx, node)
        self._emit(
            ctx,
            ToolReturned(
                session=session,
                turn=turn,
                call=node.call,
                tool=node.name,
                status=result.status.value,
                summary=result.summary,
            ),
        )
        # A body that did not fit was written to the store by the tool that produced it, before this
        # result existed, so a handle that is here names content that is already there. Saying it out
        # loud is for the person watching: what they are not seeing still exists, under this name.
        for part in result.body:
            if isinstance(part, Document):
                self._emit(ctx, DocumentStored(session=session, document=part.document_id, chars=part.chars))
        return result

    async def _ran(self, ctx: TurnCtx, node: _Call) -> ToolResult:
        """Ask about this call if its declarations say to, then run it — or answer the refusal."""
        spec, args = node.spec, node.args
        assert spec is not None, "a call with nothing to run is answered without reaching here"
        turned_down = await self._approve(ctx, node)
        if turned_down is not None:
            return turned_down
        reads, writes = _locks(node.held)
        env = Env(ctx, spec.uses, node.call)
        try:
            # The lock is the host's, so this is where two turns in two conversations agree about one file
            # — the graph above only orders the calls of a single answer.
            async with ctx.host.locks.hold(reads=reads, writes=writes):
                return await env.scope().guard(spec.run(env, args))
        except Cancelled:
            return ToolResult.of(Failure.LIMIT, "stopped")

    async def _approve(self, ctx: TurnCtx, node: _Call) -> ToolResult | None:
        """Put this call to a person when its posture says to; ``None`` when nothing has to be asked.

        A call naming a path the workspace will not resolve is let through unasked and refused by its own
        handler: asking somebody to allow a path that is refused before it is looked at would be the wrong
        question.
        """
        spec, args = node.spec, node.args
        assert spec is not None
        posture, remember = spec.approval(args)
        if posture is Posture.AUTO or node.refused:
            return None
        subjects, scoped = _subjects(ctx, spec, remember, node.held)
        approvals = ctx.host.approvals
        if not approvals.needs(posture, remember=remember, key=scoped):
            return None
        question = f"allow {spec.name}" + (f" on {subjects}" if subjects else "") + "?"
        agreed = await ctx.host.human.confirm(prompt=question, site=ctx.site(node.call, "approve"))
        if not agreed.accepted:
            return ToolResult.of(Failure.DENIED, "this call was not allowed")
        approvals.agree(posture, remember=remember, key=scoped)
        return None

    def _emit(self, ctx: TurnCtx, event: AnyEvent) -> None:
        """Say what a call is doing, through the same channel every other claim travels on."""
        ctx.host.human.emit(event)


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
    ctx: TurnCtx, spec: ToolSpec, remember: Remember | None, held: tuple[Held, ...]
) -> tuple[str, tuple[str, ...]]:
    """What an approval of this call is about: something to say, and something to remember it by.

    A path is keyed by :func:`~plyngent.kernel.paths.key` — the same normalization a lock uses — so
    ``a/../b`` and ``b`` are one approval. A session names itself, because the table belongs to the host
    and an approval in one conversation must not answer for another. A host-scoped approval covers
    everything, so its key is empty: there is only one host.
    """
    if any(role in _ACCESS for role in (spec.roles or {}).values()):
        if not held:
            return "", ()
        names = ", ".join(display(item.path, root=ctx.host.workspace.root) for item in held)
        return names, tuple(sorted(key(item.path) for item in held))
    if remember is Remember.SESSION:
        return ctx.session.session_id, (ctx.session.session_id,)
    return "", ()


def _paths(ctx: TurnCtx, spec: ToolSpec, args: object) -> tuple[tuple[Held, ...], bool]:
    """Every path this call declares, resolved with its role, and whether any of them was refused.

    Resolution is per argument rather than all-or-nothing: a call naming two paths and getting one
    refused still reads or writes the other, so what it holds is the paths it will really touch — under
    the role it declared for each, because that is what a batch orders calls by. An argument that is
    empty names no path at all: a tool with two locators leaves the other one blank, and a blank is not
    a request for the workspace root. A refused path is
    reported rather than guessed at: nothing touches a path the workspace will not resolve, so there is no
    lock to take for it, but "this call named a path and it was refused" is a different fact from "this
    call named no path", and both the approval and the batch read the difference.
    """
    workspace = ctx.host.workspace
    held: list[Held] = []
    refused = False
    for name, role in (spec.roles or {}).items():
        value = argument(args, name)
        if role not in _ACCESS or not isinstance(value, str) or not value:
            continue
        try:
            held.append(Held(workspace.resolve(value, _ACCESS[role]), role))
        except PathRefused:
            refused = True
    return tuple(held), refused


def _prepared(ctx: TurnCtx, catalog: ToolCatalog, calls: Sequence[tuple[str, Call]]) -> list[_Call]:
    """Read every call of one answer before any of them runs.

    A batch has to be decoded as a whole, because what a call will touch is inside its arguments and the
    schedule is made of that. Two answers come out of it here and not later: a name nobody has, and
    arguments that are not this tool's, are answered without running — and a call that answers without
    running touches nothing, so the schedule leaves it out.
    """
    nodes: list[_Call] = []
    for address, call in calls:
        spec = catalog.get(call.tool)
        if spec is None:
            nodes.append(
                _Call(
                    call=address,
                    name=call.tool,
                    spec=None,
                    args=None,
                    refusal=ToolResult.of(Failure.ARGS, f"no tool is named {call.tool!r}"),
                )
            )
            continue
        args, refusal = _read(spec, call.arguments)
        node = _Call(call=address, name=call.tool, spec=spec, args=args, refusal=refusal)
        if args is not None:
            node.held, node.refused = _paths(ctx, spec, args)
        nodes.append(node)
    return nodes


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
