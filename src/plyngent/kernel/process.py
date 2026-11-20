"""Running programs: one node per command, and one tree per call.

A call runs a *tree* of commands, and a single command is the tree with one node in it — which is why
this module has no separate "run one argv" entry point: the two would be two readings of the same six
fields (argv, where, environment, input, deadline, output) and they would drift.

What is *not* here matters as much as what is:

* **no policy and no human.** Where a node may run, and whether its argv needs a person, is the
  caller's answer, asked once per node through ``gate`` — lazily, before that node is spawned, because
  ``failfast`` skips nodes and asking about a command that will not run is a question with no subject.
  For the same reason no path is resolved here: ``cwd``, a redirect target and a file to read are all
  resolved by the caller, which is the only side that has a workspace.
* **no result envelope.** The engine reports what it started (:class:`Ran`) and why it stopped
  (:class:`Refused`); turning that into parts, summaries and effects is the tool layer's job.

Four rules the execution obeys:

* **A child gets a session of its own**, so stopping it kills the process *group*: ``sh -c 'sleep 300
  &'`` must not leave a ``sleep`` behind. That is the one thing about a child the kernel can still
  control after handing it the machine.
* **A non-zero exit is data, not a failure.** The result is what the program said; only "it never
  started" and "it never finished" are failures, and neither can be swallowed by ``failfast``.
* **Output is bounded twice**: per command, cut in place and marked; across the call, the call stops —
  a runaway is a runaway, and more of it helps nobody. What a pipe hands on is not bounded, because it
  is not read here: it crosses as it arrives, and the model is told how many bytes went past.
* **A pipe is a pipe.** Two programs this call named, joined by their streams — no ``&&``, no ``$VAR``,
  no glob, no word splitting. Anything a shell would do with the line is a program to name or a path to
  spell out; ``sh -c`` is the way to ask for a shell, and asking for it is what makes a person look.
"""

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import signal
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

import msgspec
from msgspec import Struct

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.failure import Failure

if TYPE_CHECKING:
    from asyncio import SubprocessTransport
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path
    from typing import Any, BinaryIO

    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.toolresult import DenialKind, Hint

#: What a node gets when it does not ask for a deadline of its own.
DEFAULT_TIMEOUT_SECONDS = 30.0
#: How many commands one call may run, however the tree nests.
DEFAULT_MAX_NODES = 20
#: How deep the continuations may nest.
MAX_DEPTH = 4
#: How much of one command's output is kept; the rest is said to exist and dropped.
NODE_OUTPUT_CAP = 32_000
#: How much output the whole call may produce before it is stopped.
OUTPUT_BUDGET = 128_000
#: How much of one pipe is taken at a time: one read, one write, then the next, so a stage that never
#: finished cannot wait for its predecessor to be over (§2 R6).
PIPE_CHUNK = 65_536
#: How long a command's pipes are still read after its group was killed. What a program said *before* a
#: deadline is usually the answer to why it hung, and killing it takes the rest with it — but a program
#: that left the group can hold a pipe open forever, so this is a wait and not a promise (§2 R6).
DRAIN_SECONDS = 0.5

#: Variables a caller may not inject: each one changes how a program *it did not write* loads code.
#: ``PYTHONPATH`` is deliberately absent — ``PYTHONPATH=src pytest`` is how a project runs its own
#: tests, and a tool that refused it would be refusing the most ordinary command there is. Removing one
#: of these is not the same act as setting it: a name that is taken away cannot change how anything
#: loads, so a null is allowed where a value is not.
DENIED_ENV: frozenset[str] = frozenset(
    {
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "GCONV_PATH",
        "LOCPATH",
        "BASH_ENV",
        "ENV",
        "IFS",
    }
)


class Stdin(Struct, frozen=True, kw_only=True, tag_field="type"):
    """Where a node's standard input comes from, when it does not come from the pipe."""


class FromText(Stdin, tag="text", frozen=True, kw_only=True):
    """A literal string, sent and then closed."""

    text: str = ""


class FromFile(Stdin, tag="file", frozen=True, kw_only=True):
    """A file's bytes, read before the command starts."""

    path: str = ""


#: Which command in a chain decides whether the chain worked.
type Failfast = bool | Literal["simple", "pipefail"]


class Execution(Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One command, and the continuations it may run.

    Every field but ``argv`` is optional, and each one inherits: a node without a ``cwd`` runs where its
    parent ran, and a node without ``failfast`` is judged the way its parent was. The call's own values
    are the outermost default, so a call that runs one command names nothing twice.

    ``pipe`` is either the next command in the chain or a file to write this one's output into — a
    string is a redirection, not a program. A node reached through a pipe is *fed*, and asking for
    another input on top of that is an error, and no input silently wins. The chain has one deadline,
    taken from the command that owns the pipe: a stage may not name its own, and neither may it branch,
    because "did this work" belongs to the command the pipe is written on.

    A pipe carries bytes between two programs this call named, and does nothing else a shell would do:
    no ``&&``, no ``$VAR``, no glob, no word splitting. Anything the line would have to be *parsed* for
    is a program to name or a path to spell out — ``sh -c`` is how to ask for a shell, and asking for
    one is what makes a person look.

    ``env`` is an overlay on the environment this program was started with: a name given a value is set,
    and a name given ``None`` is **taken away**. Taking one away is what a caller has to be able to do to
    be the author of its own command, and not the inheritor of this one's — an ``GIT_DIR`` inherited
    from the program that started this one points a command at the wrong repository, and no value a
    caller could give it says "the one you would find yourself".
    """

    argv: list[str]
    cwd: str | None = None
    env: dict[str, str | None] | None = None
    stdin: FromText | FromFile | None = None
    timeout_secs: float | None = None
    failfast: Failfast | None = None
    mix_stderr: bool = False
    then: list[Execution] = msgspec.field(default_factory=list)
    else_: list[Execution] = msgspec.field(default_factory=list, name="else")
    pipe: Execution | str | None = None

    def __post_init__(self) -> None:
        if not self.argv:
            msg = "argv must not be empty"
            raise ValueError(msg)
        for name, value in (self.env or {}).items():
            if value is not None and name.upper() in DENIED_ENV:
                msg = f"env may not set {name}: it changes how a program loads code, not what it does"
                raise ValueError(msg)
        if isinstance(self.pipe, str) and not self.pipe:
            msg = "pipe is the next command or a file to write to, and an empty name is neither"
            raise ValueError(msg)
        if self.timeout_secs is not None and self.timeout_secs <= 0:
            msg = f"timeout_secs must be positive, not {self.timeout_secs}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Permit:
    """What the caller allows for one node, already resolved.

    Three paths, because a call can name three: where the command runs, where its output is sent when
    ``pipe`` names a file, and the file its input is read from. The engine resolves none of them — it
    has no workspace, and a second resolver would be a second answer to "may this path be touched".
    """

    cwd: Path
    sink: Path | None = None
    source: Path | None = None


@dataclass(frozen=True, slots=True)
class Refused:
    """Why the call stopped before it could finish.

    ``LIMIT`` covers a cancelled call as well: a turn that was stopped is not a failure of the world,
    and it is never retried as a whole. ``hints`` is what the caller's refusal offered as the way
    through — a grant, another directory — carried here, because the message is a sentence and was never
    meant to be parsed.
    """

    failure: Failure
    message: str
    hints: tuple[Hint, ...] = ()
    #: Why, machine-readably, when the failure is a denial: this engine refuses a *command* and answers
    #: with its own value, so the kind travels here and becomes a
    #: :meth:`~plyngent.kernel.toolresult.ToolResult.denied` at the end (§0.15, E118).
    denial: DenialKind | None = None


@dataclass(frozen=True, slots=True)
class Ran:
    """One command the call actually started, and everything a result needs to say about it."""

    argv: tuple[str, ...]
    cwd: str
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    errout: str
    depth: int = 0
    via: str = ""
    fed: bool = False
    mix_stderr: bool = False
    #: Whether something outlived the kill: a process that left the group and still holds the pipes.
    #: Nothing portable can stop it, so this is what says it is there — and what a caller's result says
    #: as ``escaped=true`` (E161).
    escaped: bool = False
    #: How many bytes of this command's output went into the next program of its own chain, when it has
    #: one. Those bytes are not in ``stdout``: they are the next program's input, and the field exists so
    #: that "there were 40 000 of them" is said instead of a body that silently ended early. A count of
    #: bytes, taken before any decode, because the pipe carried bytes (§2 R6).
    handed: int | None = None

    @property
    def failed(self) -> bool:
        """Whether it did not do what it was asked, by the only measure there is: its exit."""
        return self.timed_out or self.exit_code != 0


@dataclass(frozen=True, slots=True)
class Batch:
    """What one call produced: the commands it ran, and why it may have stopped.

    Two different things stop a call and they are not the same fact. ``stopped`` is a reason it could
    not go on — a program that never started, a deadline, a budget, the turn being cancelled, a node the
    caller would not allow. ``halted`` is a command that failed where ``failfast`` says not to continue:
    the call is complete, what the command said is the answer, and nothing after it ran.
    """

    ran: tuple[Ran, ...]
    stopped: Refused | None = None
    halted: bool = False
    nodes: int = 0
    truncated: bool = False

    @property
    def started(self) -> bool:
        """Whether anything ran at all — which is what tells ``partial`` from ``failed``."""
        return bool(self.ran)

    @property
    def ended(self) -> bool:
        """Whether the call stops here, for either reason."""
        return self.stopped is not None or self.halted


#: Asked once per node, before it is spawned: allow this node as resolved, or say why it does not run.
type Gate = Callable[[Execution, Path], Awaitable[Permit | Refused]]


def validate(tree: Sequence[Execution]) -> int:
    """Refuse a tree that would not run, and say how many nodes it has.

    This checks the shape and spawns nothing: a count, a depth, and the three ways a pipe and a node
    can disagree. It runs before anything is spawned, so a call that cannot work costs nothing and asks
    nobody.
    """
    nodes = 0

    def walk(level: Sequence[Execution], *, where: str, depth: int, fed: bool) -> None:
        nonlocal nodes
        for index, node in enumerate(level):
            at = f"{where}[{index}]"
            if depth > MAX_DEPTH:
                msg = f"{at}: continuations nest at most {MAX_DEPTH} levels deep"
                raise ValueError(msg)
            nodes += 1
            if nodes > DEFAULT_MAX_NODES:
                msg = f"{at}: more than {DEFAULT_MAX_NODES} commands in one call"
                raise ValueError(msg)
            if fed and node.stdin is not None:
                msg = f"{at}: stdin comes from the pipe; drop stdin or split the command"
                raise ValueError(msg)
            if fed and (node.then or node.else_):
                msg = f"{at}: a pipe step takes no then/else; branch on the command that owns the pipe"
                raise ValueError(msg)
            if fed and node.timeout_secs is not None:
                msg = f"{at}: a pipe step takes no timeout_secs; the chain has one deadline, owned by its first command"
                raise ValueError(msg)
            walk(node.then, where=f"{at}.then", depth=depth + 1, fed=False)
            walk(node.else_, where=f"{at}.else", depth=depth + 1, fed=False)
            if isinstance(node.pipe, Execution):
                walk((node.pipe,), where=f"{at}.pipe", depth=depth + 1, fed=True)

    walk(tree, where="steps", depth=0, fed=False)
    if not nodes:
        msg = "steps must not be empty"
        raise ValueError(msg)
    return nodes


async def run(
    tree: Sequence[Execution],
    *,
    cwd: Path,
    env: Mapping[str, str | None] | None = None,
    failfast: Failfast = True,
    extra_timeout_secs: float = 0.0,
    gate: Gate,
    scope: Scope,
    budget: int = OUTPUT_BUDGET,
) -> Batch:
    """Run ``tree``, asking ``gate`` once per node before that node is spawned.

    ``extra_timeout_secs`` is the call's one knob for "this needs longer": it is added to every node's
    deadline, so a caller that has not thought about it still gets a bound, and a caller that has thinks
    about it once, not once per node.
    """
    return await _Run(
        cwd=cwd,
        env=env,
        failfast=failfast,
        extra_timeout_secs=max(0.0, extra_timeout_secs),
        gate=gate,
        scope=scope,
        budget=budget,
        nodes=validate(tree),
    ).tree(tree)


class _Run:
    """The state of one call: what ran, what it cost, and whether it stopped."""

    cwd: Path
    env: Mapping[str, str | None] | None
    failfast: Failfast
    extra_timeout_secs: float
    gate: Gate
    scope: Scope
    budget: int
    nodes: int
    ran: list[Ran]
    chars: int
    truncated: bool
    stopped: Refused | None
    halted: bool

    def __init__(
        self,
        *,
        cwd: Path,
        env: Mapping[str, str | None] | None,
        failfast: Failfast,
        extra_timeout_secs: float,
        gate: Gate,
        scope: Scope,
        budget: int,
        nodes: int,
    ) -> None:
        self.cwd = cwd
        self.env = env
        self.failfast = failfast
        self.extra_timeout_secs = extra_timeout_secs
        self.gate = gate
        self.scope = scope
        self.budget = budget
        self.nodes = nodes
        self.ran = []
        self.chars = 0
        self.truncated = False
        self.stopped = None
        self.halted = False

    async def tree(self, tree: Sequence[Execution]) -> Batch:
        """The whole call, as the value the tool layer reads."""
        await self.level(tree, depth=0, branch="", cwd=self.cwd, failfast=self.failfast)
        return Batch(
            ran=tuple(self.ran),
            stopped=self.stopped,
            halted=self.halted,
            nodes=self.nodes,
            truncated=self.truncated,
        )

    async def level(
        self, nodes: Sequence[Execution], *, depth: int, branch: str, cwd: Path, failfast: Failfast
    ) -> None:
        """One list of siblings, in the order they were given."""
        for node in nodes:
            if self.ended:
                return
            await self.node(node, depth=depth, branch=branch, cwd=cwd, failfast=failfast)

    async def node(self, node: Execution, *, depth: int, branch: str, cwd: Path, failfast: Failfast) -> None:
        """One command and the chain it owns, then the continuations its outcome selects.

        "Did this work" belongs to the command that owns the pipe, so the continuations follow the stage
        that decides — the last one, or under ``pipefail`` the first that failed, which is the one a
        message can name.
        """
        if self._cancelled():
            return
        mode = node.failfast if node.failfast is not None else failfast
        permit = await self.gate(node, cwd)
        if isinstance(permit, Refused):
            self._halt(permit)
            return
        stdin = self._stdin(node, permit)
        if isinstance(stdin, Refused):
            self._halt(stdin)
            return

        outcome = await self.pipeline(node, permit, stdin=stdin, depth=depth, branch=branch, mode=mode)
        if outcome is not None and not self.ended:
            await self.branch(node, outcome, mode, depth=depth, cwd=permit.cwd)

    async def branch(self, node: Execution, stage: Ran, mode: Failfast, *, depth: int, cwd: Path) -> None:
        """Follow the outcome: the success list, the failure list, or stop following the tree."""
        if not stage.failed:
            await self.level(node.then, depth=depth + 1, branch="then", cwd=cwd, failfast=mode)
            return
        if node.else_:
            await self.level(node.else_, depth=depth + 1, branch="else", cwd=cwd, failfast=mode)
        elif _stops(mode):
            # Not a failure of the call: what the command said *is* the answer, and this is as far as
            # the caller asked to go.
            self.halted = True

    async def pipeline(
        self, node: Execution, permit: Permit, *, stdin: bytes | None, depth: int, branch: str, mode: Failfast
    ) -> Ran | None:
        """One program and everything its ``pipe`` adds, run as one pipeline.

        The stages are spawned together and joined by a pump each, so the bytes a middle stage writes
        reach the next program as they arrive: a chain whose first program does not stop answers as soon
        as its last one does, and nothing is held to be handed over later. What a middle stage is
        answered with is the count of bytes that crossed.

        A chain has one deadline and one cancellation, both taken from the command that owns the pipe
        (``validate`` refuses either on a stage), and a stop kills every stage's process group. The
        stages are gated before anything is spawned, so a chain that cannot run costs nothing.
        """
        stages = [node, *_rest(node.pipe)]
        plans = await self._planned(stages, permit)
        if plans is None:
            return None

        sink = plans[-1].sink if isinstance(stages[-1].pipe, str) else None
        opened: BinaryIO | None = None
        if sink is not None:
            try:
                opened = _open_sink(sink)
            except OSError as exc:
                self._halt(Refused(Failure.of(exc), f"cannot write {sink}: {exc}"))
                return None

        started: list[_Stage] = []
        try:
            await self._spawned(stages, plans, started, depth=depth, branch=branch, sink=sink, opened=opened)
        except _CannotStartError as exc:
            if opened is not None:
                opened.close()
            await self._abandon(started)
            self._halt(Refused(exc.failure, exc.message))
            return None

        timeout = self._deadline(node)
        readers = self._readers(started)
        feeding = asyncio.ensure_future(_fed(started[0].process.stdin, stdin))
        working = [*readers, feeding, asyncio.ensure_future(_ended([stage.process for stage in started]))]
        running = asyncio.ensure_future(asyncio.gather(*working, return_exceptions=True))
        try:
            _ = await self.scope.guard(asyncio.wait_for(asyncio.shield(running), timeout))
        except TimeoutError:
            await self._timed_out(started, readers)
        except Cancelled:
            # The turn was stopped under this chain: the call did not fail, it was called off.
            await self._dropped(started, running)
            self._halt(Refused(Failure.LIMIT, f"stopped while running: {shlex.join(node.argv)}"))
            return None
        except asyncio.CancelledError:
            await self._dropped(started, running)
            raise
        _ = await running
        if opened is not None:
            opened.close()
        return self._published(started, node, timeout=timeout, mode=mode)

    def _deadline(self, node: Execution) -> float:
        """How long the whole chain gets: its owner's deadline, plus whatever the call added once."""
        return max(0.0, (node.timeout_secs or DEFAULT_TIMEOUT_SECONDS) + self.extra_timeout_secs)

    async def _planned(self, stages: Sequence[Execution], permit: Permit) -> list[Permit] | None:
        """Ask about every stage before any of them runs, so a chain that cannot run costs nothing."""
        plans = [permit]
        for child in stages[1:]:
            allowed = await self.gate(child, permit.cwd)
            if isinstance(allowed, Refused):
                self._halt(allowed)
                return None
            plans.append(allowed)
        return plans

    def _published(self, started: Sequence[_Stage], node: Execution, *, timeout: float, mode: Failfast) -> Ran | None:
        """Record what ran, and answer the stage that decides whether the continuations follow."""
        rans = [self._ran(stage) for stage in started]
        self.ran.extend(rans)
        self.chars += sum(len(ran.stdout) + len(ran.stderr) + len(ran.errout) for ran in rans)
        refusal = self._refused(started, node, timeout)
        if refusal is not None:
            self._halt(refusal)
            return None
        failed = next((ran for ran in rans if ran.failed), None)
        return (failed or rans[-1]) if _pipefail(mode) else rans[-1]

    async def _spawned(
        self,
        stages: Sequence[Execution],
        plans: Sequence[Permit],
        into: list[_Stage],
        *,
        depth: int,
        branch: str,
        sink: Path | None,
        opened: BinaryIO | None,
    ) -> None:
        """Start every stage, in the order each one reads the one before, and record what started.

        A stage that never started leaves the ones before it with nobody to read them, so the caller
        kills those: a chain is one program in several parts, and half of it is not what was asked for.
        """
        last = len(stages) - 1
        for index, (child, plan) in enumerate(zip(stages, plans, strict=True)):
            into.append(
                _Stage(
                    node=child,
                    cwd=plan.cwd,
                    process=await _spawn(
                        child.argv, cwd=plan.cwd, env=self._environment(child.env), mix_stderr=child.mix_stderr
                    ),
                    depth=depth + index,
                    via=branch if index == 0 else "pipe",
                    first=index == 0,
                    last=index == last,
                    sink=opened if index == last else None,
                    sink_path=sink if index == last else None,
                )
            )

    def _readers(self, stages: Sequence[_Stage]) -> list[asyncio.Future[None]]:
        """Start every reader the pipeline needs: one per stream, and one per boundary."""
        readers: list[asyncio.Future[None]] = []
        for index, stage in enumerate(stages):
            if stage.sink is not None:
                pump = asyncio.ensure_future(self._tee(stage))
            elif index + 1 < len(stages):
                pump = asyncio.ensure_future(self._hand(stage, stages[index + 1]))
            else:
                pump = asyncio.ensure_future(_capture(stage.process.stdout, stage.out))
            stage.readers.append(pump)
            readers.append(pump)
            if not stage.node.mix_stderr:
                errors = asyncio.ensure_future(_capture(stage.process.stderr, stage.err))
                stage.readers.append(errors)
                readers.append(errors)
        return readers

    async def _hand(self, stage: _Stage, next_stage: _Stage) -> None:
        """One stage's output to the next program's input, counting the bytes that crossed.

        A downstream program that closes the pipe said "enough" — ``head -1`` is the ordinary case — so
        this stops reading and closes the upstream end, which is what makes the writer meet ``EPIPE``
        instead of blocking on a pipe nobody drains. ``handed`` is bytes and not characters: the count
        is taken before anything decodes, and what a body shows of it is that number.
        """
        stream, writer = stage.process.stdout, next_stage.process.stdin
        if stream is None:
            return
        try:
            while chunk := await stream.read(PIPE_CHUNK):
                if writer is not None:
                    try:
                        writer.write(chunk)
                        await writer.drain()
                    except BrokenPipeError, ConnectionResetError:
                        _close_pipes(stage.process)
                        return
                stage.handed += len(chunk)
        finally:
            if writer is not None:
                writer.close()

    async def _tee(self, stage: _Stage) -> None:
        """A redirect: every byte of it into the file, and the start of it into the body.

        The file gets what the program wrote, whole and unchanged — redirecting is how a caller keeps an
        output too large to read here, so cutting it at the file would cut the thing it asked for. The
        body keeps a bounded prefix, because the model reads a page and not a file.
        """
        stream, sink = stage.process.stdout, stage.sink
        if stream is None or sink is None:
            return
        while chunk := await stream.read(PIPE_CHUNK):
            stage.out.take(chunk)
            try:
                _ = await self.scope.blocking(sink.write, chunk)
            except OSError as exc:
                stage.write_error = exc
                _close_pipes(stage.process)
                return

    async def _timed_out(self, stages: Sequence[_Stage], readers: Sequence[asyncio.Future[None]]) -> None:
        """Kill every group, and say which stages outlived the kill.

        ``DRAIN_SECONDS`` is spent once for the whole pipeline: the readers are waiting on the same
        kill, and what a program said before the deadline is the answer to why it never finished. A
        stage whose readers are still waiting has something outside its group holding a pipe — a
        program that called ``setsid`` — so this end of it is closed and the field says so (E161).
        """
        for stage in stages:
            stage.hung = stage.process.returncode is None
            _kill(stage.process)
        _done, pending = await asyncio.wait(readers, timeout=DRAIN_SECONDS)
        for stage in stages:
            if all(reader.done() for reader in stage.readers):
                continue
            stage.escaped = True
            _close_pipes(stage.process)
        for reader in pending:
            _ = reader.cancel()
        _ = await asyncio.gather(*readers, return_exceptions=True)

    async def _abandon(self, stages: Sequence[_Stage]) -> None:
        """Kill what did start, because a later stage of the same pipeline never did."""
        for stage in stages:
            _kill(stage.process)
            _close_pipes(stage.process)
        _ = await asyncio.gather(*(reader for stage in stages for reader in stage.readers), return_exceptions=True)

    async def _dropped(self, stages: Sequence[_Stage], running: asyncio.Future[Any]) -> None:
        """Kill what started and join everything this chain left in flight, so nothing outlives the call."""
        await self._abandon(stages)
        _ = await asyncio.gather(running, return_exceptions=True)

    def _ran(self, stage: _Stage) -> Ran:
        """One stage as the record it earns: the last one shows its streams, a middle one its count."""
        merged = stage.node.mix_stderr
        output = stage.out.bounded()
        errors = stage.err.bounded()
        self.truncated = self.truncated or stage.out.more or stage.err.more
        return Ran(
            argv=tuple(stage.node.argv),
            cwd=str(stage.cwd),
            exit_code=None if stage.hung else stage.process.returncode,
            timed_out=stage.hung,
            escaped=stage.escaped,
            handed=None if stage.last else stage.handed,
            stdout="" if merged or not stage.last else output,
            stderr="" if merged or not stage.last else errors,
            errout=output if merged and stage.last else "",
            depth=stage.depth,
            via=stage.via,
            fed=not stage.first,
            mix_stderr=merged,
        )

    def _refused(self, stages: Sequence[_Stage], node: Execution, timeout: float) -> Refused | None:
        """Why the pipeline stops the call, when it does: a failed redirect, the budget, the deadline."""
        for stage in stages:
            if stage.write_error is not None:
                return Refused(Failure.of(stage.write_error), f"cannot write {stage.sink_path}: {stage.write_error}")
        if self.chars > self.budget:
            return Refused(Failure.LIMIT, f"output budget spent ({self.budget} chars): stopped the call")
        if any(stage.hung for stage in stages):
            return Refused(Failure.TIMEOUT, f"no answer within {timeout:g}s: {shlex.join(node.argv)}")
        return None

    def _stdin(self, node: Execution, permit: Permit) -> bytes | Refused | None:
        """What this command reads, as bytes: a string, a file, or nothing — a second stage reads the pipe."""
        if isinstance(node.stdin, FromText):
            return node.stdin.text.encode()
        if isinstance(node.stdin, FromFile):
            if permit.source is None:
                msg = f"{node.argv[0]}: a file to read was given but the caller resolved none"
                return Refused(Failure.BUG, msg)
            try:
                return _read(permit.source)
            except OSError as exc:
                return Refused(Failure.of(exc), f"cannot read {permit.source}: {exc}")
        return None

    def _halt(self, refusal: Refused) -> None:
        """Stop the call here, and answer what a caller's node does when it does not run."""
        self.stopped = refusal

    @property
    def ended(self) -> bool:
        """Whether anything more should run."""
        return self.stopped is not None or self.halted

    def _cancelled(self) -> bool:
        """Stop before spawning anything else when the turn is already gone."""
        reason = self.scope.reason
        if reason is None:
            return False
        self.stopped = Refused(Failure.LIMIT, f"stopped: {reason}")
        return True

    def _environment(self, overlay: Mapping[str, str | None] | None) -> dict[str, str] | None:
        """The child's environment: this process's, the call's, then the node's — in that order.

        A name given a null is removed, because those are different environments to most programs: an
        empty ``GIT_DIR`` is a directory that does not exist, and
        no ``GIT_DIR`` at all is the repository the command would find for itself. The three layers are
        applied in order and the removal happens at the end, so a node can take away a name its call set
        and a call can take away one this program was started with.
        """
        if self.env is None and overlay is None:
            return None
        child: dict[str, str | None] = {**os.environ, **(self.env or {}), **(overlay or {})}
        return {name: value for name, value in child.items() if value is not None}


def _pipefail(mode: Failfast) -> bool:
    """Whether the whole chain decides, and not only its last command (the shell's ``pipefail``)."""
    return mode == "pipefail"


def _stops(mode: Failfast) -> bool:
    """Whether a failed command with no successor stops the call (``false`` is "keep going")."""
    return mode is not False


def _rest(pipe: Execution | str | None) -> list[Execution]:
    """The stages a chain adds after its first command, in the order each one reads the one before."""
    stages: list[Execution] = []
    while isinstance(pipe, Execution):
        stages.append(pipe)
        pipe = pipe.pipe
    return stages


def _utf8() -> codecs.IncrementalDecoder:
    """UTF-8 read as it arrives, so a byte sequence cut in half costs one replacement character."""
    return codecs.getincrementaldecoder("utf-8")("replace")


@dataclass(slots=True)
class _Caption:
    """One stream read to its end: the start of it as text, and how much of it there was.

    The cap is on characters, because a body is text the model reads; the bytes are counted apart so
    that "there was more" is a fact about the stream and not about where a decode stopped. Reading to
    the end is what keeps a command from blocking on a pipe whose output is only half kept.
    """

    text: str = ""
    seen: int = 0
    eaten: int = 0
    cut: bool = False
    decoder: codecs.IncrementalDecoder = field(default_factory=_utf8)

    def take(self, chunk: bytes) -> None:
        """One read from the pipe: kept while there is room, counted either way."""
        self.seen += len(chunk)
        if len(self.text) >= NODE_OUTPUT_CAP:
            return
        self.eaten += len(chunk)
        self.text += self.decoder.decode(chunk)
        if len(self.text) > NODE_OUTPUT_CAP:
            self.text = self.text[:NODE_OUTPUT_CAP]
            self.cut = True

    @property
    def more(self) -> bool:
        """Whether more arrived than the text holds — which is what says a body was cut."""
        return self.cut or self.seen > self.eaten

    def bounded(self) -> str:
        """What a body carries: the text that was kept, marked when there was more of it."""
        if not self.more:
            return self.text
        return f"{self.text}\n...[truncated at {NODE_OUTPUT_CAP} chars]"


@dataclass(slots=True)
class _Stage:
    """One command of a pipeline: what it was asked to do, and what came back.

    ``handed`` is the count a middle stage is answered with: its output went into the next program, so
    those bytes are not in the body and the number says there were that many. ``hung`` is set when the
    chain's one deadline found this one running, which is what separates "it never answered" from "it
    answered before the deadline found the chain".
    """

    node: Execution
    cwd: Path
    process: asyncio.subprocess.Process
    depth: int
    via: str
    first: bool
    last: bool
    out: _Caption = field(default_factory=_Caption)
    err: _Caption = field(default_factory=_Caption)
    readers: list[asyncio.Future[None]] = field(default_factory=list)
    handed: int = 0
    hung: bool = False
    escaped: bool = False
    #: The file a ``pipe`` string redirected this stage's output into, when it is the last stage, and the
    #: path it is, so a write that fails can name the file the call asked for.
    sink: BinaryIO | None = None
    sink_path: Path | None = None
    #: Why the redirect stopped: the file was opened, and writing to it failed later.
    write_error: OSError | None = None


class _CannotStartError(Exception):
    """The child never existed: the executable is missing, or the system refused to start it."""

    failure: Failure
    message: str

    def __init__(self, failure: Failure, message: str) -> None:
        super().__init__(message)
        self.failure = failure
        self.message = message


async def _capture(stream: asyncio.StreamReader | None, into: _Caption) -> None:
    """One pipe, in chunks, to its end — so a kill leaves behind what had already arrived."""
    if stream is None:
        return
    while chunk := await stream.read(PIPE_CHUNK):
        into.take(chunk)


async def _ended(processes: Sequence[asyncio.subprocess.Process]) -> None:
    """Every stage stopped, by its own exit or by the kill that ended the one deadline."""
    _ = await asyncio.gather(*(process.wait() for process in processes))


async def _fed(stream: asyncio.StreamWriter | None, stdin: bytes | None) -> None:
    """Hand the child its input and close the pipe, which is what makes a reader see its end.

    A child that never read what it was sent closes the pipe under the write, and that is a fact about
    the command, and the call did not fail.
    """
    if stream is None:
        return
    try:
        if stdin:
            stream.write(stdin)
            await stream.drain()
    except BrokenPipeError, ConnectionResetError:
        pass
    finally:
        stream.close()


def _close_pipes(process: asyncio.subprocess.Process) -> None:
    """Close this end of a command's pipes, keeping what they had already carried.

    The readers see the end of the stream at once — what arrived is already in the caller's buffers —
    and the fd is released, so no transport nobody will ever finish keeps it. Whatever still holds the
    write end then meets ``EPIPE`` the next time it says something, or ``SIGPIPE`` if it did not ignore
    one. That is the closest thing to stopping it a portable program has, and it is not a kill: a program
    that never writes again keeps running, which is why the result also *says* this happened (E161).
    """
    # The public face of a subprocess has no "close my ends": `Process` exposes kill, terminate and wait,
    # and the transport is where the pipes are. `close()` on it is documented API; the attribute that holds
    # it is the private part, and this is the one place in the tree that reads it.
    transport = cast("SubprocessTransport", getattr(process, "_transport"))  # noqa: B009 — a private name cannot be written out
    transport.close()


async def _spawn(
    argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None, mix_stderr: bool
) -> asyncio.subprocess.Process:
    """Start one command in a session of its own, so its whole group can be stopped later."""
    try:
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=None if env is None else dict(env),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT if mix_stderr else asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except FileNotFoundError:
        raise _CannotStartError(Failure.MISSING, f"executable not found: {argv[0]!r}") from None
    except OSError as exc:
        raise _CannotStartError(Failure.of(exc), f"failed to start command: {exc}") from exc


def _kill(process: asyncio.subprocess.Process) -> None:
    """Kill the child's whole process group, and wait for nothing.

    The group, not the child: a shell that started something in the background is still that something's
    parent by session, and killing only the shell leaves behind exactly what the caller was trying to
    stop. Windows has no group to signal, so there the child is what gets killed.

    Nothing is waited for here, and that is the point: ``Process.wait()`` resolves only after every pipe
    is disconnected, so with something holding a write end it waits for *that* — the one process this
    cannot kill. The reaping is asyncio's own watcher, which knows nothing about pipes (E161).
    """
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
    except ProcessLookupError, PermissionError:
        _ = process.kill()


def _read(path: Path) -> bytes:
    """A file's bytes, read off the loop."""
    return path.read_bytes()


def _open_sink(path: Path) -> BinaryIO:
    """The file a redirect writes into, opened for the bytes a program actually produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("wb")
