"""One call's gate: where a node runs, what it may touch, and whether it has to be agreed to.

Every tool that starts a program asks the same three questions of every argv it starts, and they are asked
in the same order: the workspace resolves where the command runs and what its redirects touch, the command
policy judges the argv, and a person is asked about whatever the policy could not settle. So the gate is
one piece rather than a paragraph copied into each tool: ``execute`` walks a tree of them, a tool whose
whole job is one read-only program (the vcs group) hands the same engine one node, and a live terminal
asks through :func:`agreed` too — the same question, the same table of approvals, and therefore the same
yes remembered for the rest of the host.

Nothing here reads the arguments of a call to decide how dangerous it is. What a node may do is resolved
by the workspace, which is the only place that can refuse a path; what it costs to ask is the policy's
answer, and the two are joined here because this is the only place that sees both.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from plyngent.kernel.access import AccessMode
from plyngent.kernel.command import Allowed, NeedsHuman, NeedsReview
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import FromFile, Permit, Refused
from plyngent.kernel.toolresult import DenialKind, denial_says, kind_of
from plyngent.kernel.tools import lifting_hints
from plyngent.kernel.workspace import Denied, PathRefused
from plyngent.wire.ask import Agreed, Confirmation, Severity, as_agreed

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
        asked = await agreed(self.env, argv)
        if isinstance(asked, Refused):
            return asked
        self.approvals.extend(asked)
        return None


async def agreed(env: Env, argv: list[str], *, where: str = "") -> Refused | tuple[str, ...]:
    """Ask whatever this argv needs a person's yes for, and answer the programs a yes may be kept for.

    One question for a command the denylist names — asked program by program, because that is what a yes
    is remembered by — and one question for a command this build cannot answer for, which is asked every
    time because nothing about it is certain enough to keep. When both hold there is one question and no
    memory: a yes to a program whose name is a guess is not a yes to that program.

    ``where`` is what the call is doing it in — ``"on a terminal"`` — because the same argv is a different
    question when it is about to own a screen.
    """
    verdict = env.workspace.commands.verdict(argv)
    if isinstance(verdict, Allowed):
        return ()
    if isinstance(verdict, NeedsReview) or verdict.reasons:
        answer = await _asked(env, _review_prompt(argv, verdict, where=where))
        return () if answer.accepted else _refusal(answer, about=f"running {shlex.join(argv)}")
    remembered: list[str] = []
    for program in verdict.programs:
        answer = await _asked(env, _denylist_prompt(program, where=where))
        if not answer.accepted:
            return _refusal(answer, about=f"running {program!r}")
        env.workspace.commands.approve(program)
        remembered.append(program)
    return tuple(remembered)


async def _asked(env: Env, prompt: str) -> Agreed:
    """Put one question to a person, at the level a program is, and answer what they said."""
    return as_agreed(await env.ask(Confirmation(prompt=prompt, severity=Severity.DANGER), at="gate"))


def _denylist_prompt(program: str, *, where: str) -> str:
    """The question about one denylisted program, saying what the call is doing it in."""
    place = f" {where}" if where else ""
    return f"run {program!r}{place}? it is on the command denylist"


def _review_prompt(argv: list[str], verdict: NeedsHuman | NeedsReview, *, where: str) -> str:
    """The question about a whole command line, saying what could not be judged and what else is on it."""
    place = f" {where}" if where else ""
    says = (
        "; ".join(reason.says() for reason in verdict.reasons)
        if isinstance(verdict, NeedsReview)
        else f"{'; '.join(reason.says() for reason in verdict.reasons)} — and {', '.join(map(repr, verdict.programs))} "
        "is on the command denylist"
    )
    return f"run {shlex.join(argv)}{place}? {says}"


def _refusal(answer: Agreed, *, about: str) -> Refused:
    """A no, as the engine's refusal: the answer's author is the kind of denial, and nothing ran."""
    kind = kind_of(answer)
    return Refused(Failure.DENIED, denial_says(kind, about=about, policy=answer.policy), denial=kind)


def _refused(exc: PathRefused) -> Refused:
    """A path refusal as the engine's answer: the denylist is final, a missing grant is not."""
    final = isinstance(exc, Denied)
    return Refused(
        Failure.DENIED if final else Failure.BLOCKED,
        str(exc),
        lifting_hints(exc),
        denial=DenialKind.NEVER_ALLOWED if final else None,
    )
