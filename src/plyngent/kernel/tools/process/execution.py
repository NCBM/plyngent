"""The tool that runs programs: one call, one tree, and the parts a reader needs.

The engine (:mod:`plyngent.kernel.process`) knows how to run a tree and nothing about policy; this
module is the other half. Two of the three questions a command raises are answered by
:class:`~plyngent.kernel.tools.process.gate.CommandGate` — where a node may run, and whether its argv
has to be agreed to — because the tool that runs one read-only program asks exactly the same two. What
is left here is the third, and the one only a tree has:

* **What does the caller see?** One :class:`~plyngent.kernel.toolresult.ProcessControl` per command that
  actually started, in the order they started, plus the effects they caused. A program exiting non-zero
  is in that record, not in the status: the call did what it was asked.

What this layer declares is the paths inside the tree: ``holds`` walks the nodes and names the file each
one reads and the file a ``pipe`` redirects into, because ``roles`` is a table of argument names and a
tree has none to name. Those are the two files the gate resolves per node, so what a batch locks and
what a command touches are one list — a directory a command runs in is deliberately not among them, since
a lock is per path and ``sub`` says nothing about ``sub/x.txt``. What is still not declarable is the paths
a *program* chooses while it runs: a call that spells no path and lets its argv decide is a call the
scheduler cannot see, which is why two such calls that depend on each other are two answers.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.failure import Failure
from plyngent.kernel.paths import display
from plyngent.kernel.process import Batch, Execution, Failfast, FromFile, Ran, validate
from plyngent.kernel.process import run as run_tree
from plyngent.kernel.toolresult import Effects, Hint, Meta, ProcessControl, ToolResult
from plyngent.kernel.tools import lifted
from plyngent.kernel.tools.process.gate import CommandGate
from plyngent.kernel.toolspec import Group, Mutation, Reach, Res, Role, ToolArgs, ToolSpec
from plyngent.kernel.workspace import PathRefused

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from plyngent.kernel.env import Env
    from plyngent.kernel.workspace import Workspace


class ExecuteArgs(ToolArgs, kw_only=True):
    """Run commands (no shell), each with optional continuations; one command is a tree of one node.

    ``steps`` is a list of nodes, and a node is::

        {
          "argv": ["pytest", "-q"],       # required: a program and its arguments, never a shell line
          "cwd": ".",                     # optional; else the parent's, else the call's
          "env": {"FOO": "1"},            # overlay: a value sets a name, a null takes one away
          "stdin": {"type": "text", "text": "..."},   # or {"type": "file", "path": "..."}
          "timeout_secs": 30,             # optional; else 30, plus extra_timeout_secs
          "failfast": true,               # true/"simple", "pipefail", or false
          "mix_stderr": false,            # merge the two streams at the OS level
          "then":  [ ...nodes... ],       # runs when this command's exit is 0
          "else":  [ ...nodes... ],       # runs when it is not
          "pipe": { ...node... }          # takes this command's stdout as its stdin
        }

    A ``pipe`` may also be a file name, which redirects the output there and starts no other
    command. A continuation runs where its parent ran and inherits its environment and ``failfast``;
    the call's ``cwd``, ``env`` and ``failfast`` are the outermost default. ``failfast`` reads as a
    shell does: ``simple`` (the same as ``true``) judges a chain by its last command, ``pipefail`` by
    every stage of it, and ``false`` keeps going after a non-zero exit. None of them keeps going past a
    program that never started or a deadline: there is no exit code to branch on, so the call stops.

    A non-zero exit is a result, not an error: the exit code, both streams and whether it timed out are
    reported per command. Use ``then``/``else`` for what should happen next, and ``extra_timeout_secs``
    once for a batch that needs longer than the usual deadline. At most 20 commands and 4 levels of
    continuation.
    """

    steps: list[Execution]
    cwd: str = "."
    env: dict[str, str | None] | None = None
    failfast: Failfast = True
    extra_timeout_secs: float = 0.0


async def execute(env: Env, args: ExecuteArgs) -> ToolResult:
    """Run the tree, asking about each node as it comes up."""
    workspace = env.workspace
    started = time.monotonic()
    try:
        count = validate(args.steps)
    except ValueError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    try:
        base = env.resolve(args.cwd, AccessMode.EXEC)
    except PathRefused as exc:
        return lifted(exc)
    if not base.is_dir():
        return ToolResult.of(Failure.MISSING, f"not a directory: {args.cwd}")

    gate = CommandGate(workspace, env)
    try:
        batch = await run_tree(
            args.steps,
            cwd=base,
            env=args.env,
            failfast=args.failfast,
            extra_timeout_secs=args.extra_timeout_secs,
            gate=gate,
            scope=env.scope("proc-0"),
        )
    except Cancelled:
        # The engine turns a stopped turn into a refusal of its own; reaching here means the cancel
        # arrived between two of its checks, so nothing has run since the last one it saw.
        return ToolResult.of(Failure.LIMIT, "stopped")

    elapsed = int((time.monotonic() - started) * 1000)
    return _result(batch, workspace=workspace, approvals=gate.approvals, nodes=count, elapsed_ms=elapsed)


def _result(batch: Batch, *, workspace: Workspace, approvals: list[str], nodes: int, elapsed_ms: int) -> ToolResult:
    """One record per command that ran, and the status this call's ending earns."""
    body = tuple(_part(ran, workspace=workspace) for ran in batch.ran)
    effects = Effects(
        external=tuple(f"ran: {shlex.join(ran.argv)}" for ran in batch.ran),
        approvals=tuple(approvals),
    )
    meta = Meta(elapsed_ms=elapsed_ms, truncated=batch.truncated)
    summary = f"{len(batch.ran)} of {nodes} commands ran"
    if batch.stopped is not None:
        hints = batch.stopped.hints or _hints(batch)
        if batch.ran:
            return ToolResult.partial(
                f"{summary}: {batch.stopped.message}",
                effects=effects,
                body=body,
                hints=hints,
                meta=meta,
            )
        if batch.stopped.denial is not None:
            return ToolResult.denied(batch.stopped.denial, batch.stopped.message, body=body, meta=meta)
        return ToolResult.of(batch.stopped.failure, batch.stopped.message, body=body, hints=hints, meta=meta)
    if not batch.ran:
        return ToolResult.of(Failure.BUG, "a call that ran nothing did not say why", meta=meta)
    return ToolResult.ok(summary, body=body, effects=effects, meta=meta)


def _part(ran: Ran, *, workspace: Workspace) -> ProcessControl:
    """One started command, as a short piece a reader takes in at once."""
    return ProcessControl(
        argv=ran.argv,
        cwd=display(Path(ran.cwd), root=workspace.root),
        exit_code=ran.exit_code,
        timed_out=ran.timed_out,
        escaped=ran.escaped,
        fed=ran.fed,
        depth=ran.depth,
        via=ran.via,
        mix_stderr=ran.mix_stderr,
        stdout=ran.stdout,
        stderr=ran.stderr,
        errout=ran.errout,
    )


def _hints(batch: Batch) -> tuple[Hint, ...]:
    """The one argument that would have let this call finish, when there is one."""
    if batch.stopped is None or batch.stopped.failure is not Failure.TIMEOUT:
        return ()
    return (
        Hint(
            tool=EXECUTE.name,
            given=(("extra_timeout_secs", "60"),),
            why="this needs longer than the deadline it was given",
        ),
    )


def _reached(args: ExecuteArgs) -> tuple[Reach, ...]:
    """Every file this call's tree names: what each node reads as its input, and where one writes.

    ``roles`` can name the call's own ``cwd`` and nothing deeper, so this is the rest of it — and it is
    only the two files, because a directory a command runs in is not a thing two calls have to take turns
    over: a lock is per path and not per tree, so holding ``sub`` would say nothing about ``sub/x.txt``.
    These two are the paths the gate resolves for a node as well, so the walk here and the walk there see
    the same nodes, and a path that is locked and approved is a path the command will really use (§10.12).

    Nothing is resolved here: these are the values as the model spelled them, and the executor resolves
    them the way it resolves every other declared path.
    """
    reached: list[Reach] = []
    for node in _nodes(args.steps):
        if isinstance(node.stdin, FromFile):
            reached.append(Reach(value=node.stdin.path, role=Role.READ))
        if isinstance(node.pipe, str):
            reached.append(Reach(value=node.pipe, role=Role.WRITE))
    return tuple(reached)


def _nodes(steps: Sequence[Execution]) -> Iterator[Execution]:
    """Every node of a tree, in the order the model wrote them: a command, then what follows it."""
    for node in steps:
        yield node
        yield from _nodes(node.then)
        yield from _nodes(node.else_)
        if isinstance(node.pipe, Execution):
            yield from _nodes((node.pipe,))


EXECUTE = ToolSpec(
    name="execute",
    group=Group.PROCESS,
    args=ExecuteArgs,
    run=execute,
    uses=Res.EXEC | Res.PATHS,
    # A child changes what the kernel cannot see, and a redirect writes into the workspace: both are
    # declared, and the baseline still reads `AUTO` because `EXEC` is consulted before `OUTSIDE`.
    mutates=Mutation.OUTSIDE | Mutation.WORKSPACE,
    roles={"cwd": Role.CWD},
    holds=_reached,
)

SPECS: tuple[ToolSpec, ...] = (EXECUTE,)
