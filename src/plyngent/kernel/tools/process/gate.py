"""One call's gate: where a node runs, what it may touch, and whether it has to be agreed to.

Every tool that starts a program asks the same three questions of every argv it starts, and they are asked
in the same order: the workspace resolves where the command runs and what its redirects touch, the command
policy judges the argv, and a person is asked about the programs the policy names. So the gate is one
piece rather than a paragraph copied into each tool: ``execute`` walks a tree of them, and a tool whose
whole job is one read-only program (the vcs group) hands the same engine one node and gets the same
answers — the same question, the same table of approvals, and therefore the same yes remembered for the
rest of the host.

Nothing here reads the arguments of a call to decide how dangerous it is. What a node may do is resolved
by the workspace, which is the only place that can refuse a path; what it costs to ask is the policy's
answer, and the two are joined here because this is the only place that sees both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.command import NeedsHuman
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import FromFile, Permit, Refused
from plyngent.kernel.toolresult import DenialKind, denial_says, kind_of
from plyngent.kernel.tools import lifting_hints
from plyngent.kernel.workspace import Denied, PathRefused
from plyngent.wire.ask import Severity

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.env import Env
    from plyngent.kernel.process import Execution
    from plyngent.kernel.workspace import Workspace


class CommandGate:
    """One call's gate: where a node runs, what it may touch, and whether it has to be agreed to.

    The approvals are kept here rather than read back out of the command policy: the policy remembers
    programs for the life of the host, and the effect record is about *this* call.
    """

    workspace: Workspace
    env: Env
    approvals: list[str]

    def __init__(self, workspace: Workspace, env: Env) -> None:
        self.workspace = workspace
        self.env = env
        self.approvals = []

    async def __call__(self, node: Execution, inherited: Path) -> Permit | Refused:
        """Where this node runs and what it may touch, or why it does not run."""
        here = node.cwd if node.cwd is not None else str(inherited)
        try:
            cwd = self.env.resolve(here, AccessMode.EXEC)
        except PathRefused as exc:
            return _refused(exc)
        if not cwd.is_dir():
            return Refused(Failure.MISSING, f"not a directory: {here}")
        try:
            sink = self.env.resolve(node.pipe, AccessMode.WRITE) if isinstance(node.pipe, str) else None
            source = self.env.resolve(node.stdin.path, AccessMode.READ) if isinstance(node.stdin, FromFile) else None
        except PathRefused as exc:
            return _refused(exc)
        refused = await self._allowed(node.argv)
        if refused is not None:
            return refused
        return Permit(cwd=cwd, sink=sink, source=source)

    async def _allowed(self, argv: list[str]) -> Refused | None:
        """Whether this argv may run — from the policy, and from the human when the policy asks."""
        verdict = self.workspace.commands.verdict(argv)
        if not isinstance(verdict, NeedsHuman):
            return None
        for program in verdict.programs:
            agreed = await self.env.human.confirm(
                prompt=f"run {program!r}? it is on the command denylist",
                severity=Severity.DANGER,
                site=self.env.site("gate"),
            )
            if not agreed.accepted:
                kind = kind_of(agreed)
                return Refused(
                    Failure.DENIED,
                    denial_says(kind, about=f"running {program!r}", policy=agreed.policy),
                    denial=kind,
                )
            self.workspace.commands.approve(program)
            self.approvals.append(program)
        return None


def _refused(exc: PathRefused) -> Refused:
    """A path refusal as the engine's answer: the denylist is final, a missing grant is not."""
    final = isinstance(exc, Denied)
    return Refused(
        Failure.DENIED if final else Failure.BLOCKED,
        str(exc),
        lifting_hints(exc),
        denial=DenialKind.NEVER_ALLOWED if final else None,
    )
