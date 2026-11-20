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

Three rules the execution obeys:

* **A child gets a session of its own**, so stopping it kills the process *group*: ``sh -c 'sleep 300
  &'`` must not leave a ``sleep`` behind. That is the one thing about a child the kernel can still
  control after handing it the machine.
* **A non-zero exit is data, not a failure.** The result is what the program said; only "it never
  started" and "it never finished" are failures, and neither can be swallowed by ``failfast``.
* **Output is bounded twice**: per command, cut in place and marked; across the call, the call stops —
  a runaway is a runaway, and more of it helps nobody.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import msgspec
from msgspec import Struct

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.failure import Failure

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from pathlib import Path

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
#: How much output the whole call may produce before it is stopped rather than truncated.
OUTPUT_BUDGET = 128_000
#: How much of one pipe is taken at a time while a command runs.
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
    another input on top of that is an error rather than a silent winner.

    ``env`` is an overlay on the environment this program was started with: a name given a value is set,
    and a name given ``None`` is **taken away**. Taking one away is what a caller has to be able to do to
    be the author of its own command rather than the inheritor of this one's — an ``GIT_DIR`` inherited
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
    through — a grant, another directory — carried here rather than rebuilt from the message, which is
    a sentence and was never meant to be parsed.
    """

    failure: Failure
    message: str
    hints: tuple[Hint, ...] = ()
    #: Why, machine-readably, when the failure is a denial: this engine refuses a *command* and answers
    #: with its own value rather than a result, so the kind travels here and becomes a
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

    Shape rather than execution: a count, a depth, and the two ways a pipe and an input can disagree.
    It runs before anything is spawned, so a call that cannot work costs nothing and asks nobody.
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
    deadline rather than replacing it, so a caller that has not thought about it still gets a bound, and
    a caller that has thinks about it once instead of per node.
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
            _ = await self.node(node, depth=depth, branch=branch, cwd=cwd, failfast=failfast)

    async def node(
        self,
        node: Execution,
        *,
        depth: int,
        branch: str,
        fed: str | None = None,
        stage_only: bool = False,
        cwd: Path,
        failfast: Failfast,
    ) -> Ran | None:
        """One command, its chain, and the continuations its outcome selects.

        ``stage_only`` marks a pipe stage: it hands its output on and does not branch, because "did this
        work" belongs to the command that owns the pipe.
        """
        if self._cancelled():
            return None
        mode = node.failfast if node.failfast is not None else failfast
        permit = await self.gate(node, cwd)
        if isinstance(permit, Refused):
            return self._halt(permit)
        stdin = self._stdin(node, permit, fed)
        if isinstance(stdin, Refused):
            return self._halt(stdin)

        ran, refusal = await self.command(
            node, cwd=permit.cwd, stdin=stdin, depth=depth, via=branch, fed=fed is not None
        )
        if ran is None or refusal is not None:
            self.stopped = refusal
            return ran
        if permit.sink is not None:
            written = await self._write(permit.sink, ran.stdout)
            if written is not None:
                return self._halt(written)

        outcome = await self.chain(node, ran, mode, depth=depth, cwd=permit.cwd)
        if not stage_only and outcome is not None and not self.ended:
            await self.branch(node, outcome, mode, depth=depth, cwd=permit.cwd)
        return ran

    async def chain(self, node: Execution, ran: Ran, mode: Failfast, *, depth: int, cwd: Path) -> Ran | None:
        """Run the pipe chain this command owns, and answer the stage whose exit decides the outcome.

        ``simple`` (and ``false``) read the *last* command of the chain, which is what a shell does by
        default; ``pipefail`` reads every stage, and the answer is the first one that failed so the
        message can name it.
        """
        stage = ran
        failed = ran if ran.failed else None
        piped = ran.errout if ran.mix_stderr else ran.stdout
        child = node.pipe
        while isinstance(child, Execution):
            stage = await self.node(
                child, depth=depth + 1, branch="pipe", fed=piped, stage_only=True, cwd=cwd, failfast=mode
            )
            if stage is None:
                return None
            failed = failed if failed is not None else (stage if stage.failed else None)
            piped = stage.errout if stage.mix_stderr else stage.stdout
            child = child.pipe
        return (failed or stage) if _pipefail(mode) else stage

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

    async def command(
        self, node: Execution, *, cwd: Path, stdin: bytes | None, depth: int, via: str, fed: bool
    ) -> tuple[Ran | None, Refused | None]:
        """Spawn one command and collect it; a refusal when it never started or never finished."""
        timeout = max(0.0, (node.timeout_secs or DEFAULT_TIMEOUT_SECONDS) + self.extra_timeout_secs)
        merged = node.mix_stderr
        try:
            process = await _spawn(node.argv, cwd=cwd, env=self._environment(node.env), mix_stderr=merged)
        except _CannotStartError as exc:
            return None, Refused(exc.failure, exc.message)

        try:
            out, err, timed_out = await self._wait(process, stdin=stdin, wait_seconds=timeout)
        except Cancelled:
            await _stop(process)
            return None, Refused(Failure.LIMIT, f"stopped while running: {shlex.join(node.argv)}")

        output, cut = _bounded(out.decode(errors="replace"))
        errors, cut_err = _bounded(err.decode(errors="replace"))
        self.truncated = self.truncated or cut or cut_err
        ran = Ran(
            argv=tuple(node.argv),
            cwd=str(cwd),
            exit_code=None if timed_out else process.returncode,
            timed_out=timed_out,
            stdout="" if merged else output,
            stderr="" if merged else errors,
            errout=output if merged else "",
            depth=depth,
            via=via,
            fed=fed,
            mix_stderr=merged,
        )
        self.ran.append(ran)
        self.chars += len(output) + len(errors)
        if self.chars > self.budget:
            return ran, Refused(Failure.LIMIT, f"output budget spent ({self.budget} chars): stopped the call")
        if timed_out:
            return ran, Refused(Failure.TIMEOUT, f"no answer within {timeout:g}s: {shlex.join(node.argv)}")
        return ran, None

    async def _wait(
        self, process: asyncio.subprocess.Process, *, stdin: bytes | None, wait_seconds: float
    ) -> tuple[bytes, bytes, bool]:
        """Wait for the child, the deadline, or the turn — whichever arrives first.

        The pipes are read here rather than by ``communicate``, and the deadline is why: a cancelled
        ``communicate`` takes what it had already read with it, so a command that said why it was hanging
        came back as one that had said nothing. These readers are shielded from that cancellation and are
        given a moment past the kill to finish, so a timeout answers with what the command had said up to
        then — the lines are the answer to "why did it never finish" far more often than silence is.
        """
        output = bytearray()
        errors = bytearray()
        reading = asyncio.ensure_future(_collected(process, stdin=stdin, output=output, errors=errors))
        try:
            await self.scope.guard(asyncio.wait_for(asyncio.shield(reading), wait_seconds))
        except TimeoutError:
            await _stop(process)
            await _salvaged(reading)
            return bytes(output), bytes(errors), True
        except Cancelled, asyncio.CancelledError:
            _ = reading.cancel()
            _ = await asyncio.gather(reading, return_exceptions=True)
            raise
        return bytes(output), bytes(errors), False

    def _stdin(self, node: Execution, permit: Permit, fed: str | None) -> bytes | Refused | None:
        """What this command reads, as bytes: the pipe's output, a string, a file, or nothing."""
        if fed is not None:
            return fed.encode()
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

        A name given a null is removed rather than set to nothing, because those are different
        environments to most programs: an empty ``GIT_DIR`` is a directory that does not exist, and
        no ``GIT_DIR`` at all is the repository the command would find for itself. The three layers are
        applied in order and the removal happens at the end, so a node can take away a name its call set
        and a call can take away one this program was started with.
        """
        if self.env is None and overlay is None:
            return None
        child: dict[str, str | None] = {**os.environ, **(self.env or {}), **(overlay or {})}
        return {name: value for name, value in child.items() if value is not None}

    async def _write(self, sink: Path, text: str) -> Refused | None:
        """Put a command's output where the call asked for it.

        On this call's scope's books, like every blocking call here: the write is in a thread, and a
        stop that took the waiting away has not taken the writing away.
        """
        try:
            await self.scope.blocking(_replace, sink, text)
        except OSError as exc:
            return Refused(Failure.of(exc), f"cannot write {sink}: {exc}")
        return None


def _pipefail(mode: Failfast) -> bool:
    """Whether the whole chain decides, rather than only its last command (the shell's ``pipefail``)."""
    return mode == "pipefail"


def _stops(mode: Failfast) -> bool:
    """Whether a failed command with no successor stops the call (``false`` is "keep going")."""
    return mode is not False


def _bounded(text: str) -> tuple[str, bool]:
    """One command's stream, cut at the kernel's own cap and marked when it was."""
    if len(text) <= NODE_OUTPUT_CAP:
        return text, False
    return f"{text[:NODE_OUTPUT_CAP]}\n...[truncated at {NODE_OUTPUT_CAP} chars]", True


class _CannotStartError(Exception):
    """The child never existed: the executable is missing, or the system refused to start it."""

    failure: Failure
    message: str

    def __init__(self, failure: Failure, message: str) -> None:
        super().__init__(message)
        self.failure = failure
        self.message = message


async def _collected(
    process: asyncio.subprocess.Process, *, stdin: bytes | None, output: bytearray, errors: bytearray
) -> None:
    """Everything the child writes, into buffers that outlive a cut-short wait, and then its exit."""
    reading = [_read_into(process.stdout, output)]
    if process.stderr is not None:
        reading.append(_read_into(process.stderr, errors))
    # Fed and read at the same time, which is what ``communicate`` did: a command that answers a large
    # input would deadlock on two full pipes if this handed the whole of it over before reading.
    _ = await asyncio.gather(*reading, _fed(process.stdin, stdin), return_exceptions=True)
    _ = await process.wait()


async def _read_into(stream: asyncio.StreamReader | None, into: bytearray) -> None:
    """One pipe, in chunks, to its end — so a kill leaves behind what had already arrived."""
    if stream is None:
        return
    while chunk := await stream.read(PIPE_CHUNK):
        into += chunk


async def _fed(stream: asyncio.StreamWriter | None, stdin: bytes | None) -> None:
    """Hand the child its input and close the pipe, which is what makes a reader see its end.

    A child that never read what it was sent closes the pipe under the write, and that is a fact about
    the command rather than a failure of the call.
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


async def _salvaged(reading: asyncio.Future[None]) -> None:
    """Give the readers a moment past the kill, then take the buffers as they stand."""
    try:
        _ = await asyncio.wait_for(asyncio.shield(reading), DRAIN_SECONDS)
    except TimeoutError:
        _ = reading.cancel()
        _ = await asyncio.gather(reading, return_exceptions=True)


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


async def _stop(process: asyncio.subprocess.Process) -> None:
    """Kill the child's whole process group, then reap it.

    The group, not the child: a shell that started something in the background is still that
    something's parent by session, and killing only the shell leaves behind exactly what the caller was
    trying to stop. Windows has no group to signal, so there the child is what gets killed.
    """
    if process.returncode is None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError, PermissionError:
            _ = process.kill()
    try:
        _ = await process.wait()
    except ProcessLookupError:  # pragma: no cover — the child went away between the two calls
        return


def _read(path: Path) -> bytes:
    """A file's bytes, read off the loop."""
    return path.read_bytes()


def _replace(path: Path, text: str) -> None:
    """Write *text* to *path*, creating what has to exist first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(text, encoding="utf-8")
