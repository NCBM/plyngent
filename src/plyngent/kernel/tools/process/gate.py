"""One call's gate: where a node runs, what it may touch, and whether it has to be agreed to.

Every tool that starts a program asks the same three questions of every argv it starts, and they are asked
in the same order: the workspace resolves where the command runs and what its redirects touch, the command
policy judges the argv, and a person is asked about whatever the policy could not settle. So the gate is
one piece, and no tool carries its own copy of that paragraph: ``execute`` walks a tree of them, a tool whose
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
from plyngent.kernel.command_scan import REVIEW_SAYS, Code, Reason, Review, code_of, is_interpreter, unwrap_command
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import FromFile, FromText, Permit, Refused
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

    The approvals are kept here, and are never read back out of the command policy: the policy remembers
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
        refused = await self._allowed(node.argv, code=_code(node, self.env, cwd))
        if refused is not None:
            return refused
        return Permit(cwd=cwd, sink=sink, source=source)

    async def _allowed(self, argv: list[str], *, code: Code) -> Refused | None:
        """Whether this argv may run — from the policy, and from the human when the policy asks."""
        asked = await agreed(self.env, argv, code=code)
        if isinstance(asked, Refused):
            return asked
        self.approvals.extend(asked)
        return None


def _code(node: Execution, env: Env, cwd: Path) -> Code:
    """The text this node is about to run, when its own command line says it runs text.

    One answer for every way an interpreter is handed a program: the code on its line, the file it names,
    the file its input comes from, the module it finds on its own path. A node that runs no interpreter
    answers with nothing, and so does one whose command line this build cannot read — the question about
    that one says so itself (:func:`~plyngent.kernel.command_scan.code_of`).
    """
    scan = unwrap_command(node.argv)
    if not is_interpreter(scan.base):
        return Code()
    code = code_of(scan.command)
    if code.text or code.says:
        return code
    if code.path:
        return _script(code.path, env, cwd=cwd)
    match node.stdin:
        case FromText(text=text):
            return Code(text=text)
        case FromFile(path=path):
            return _script(path, env, cwd=None)
        case _:
            return Code()


def _script(path: str, env: Env, *, cwd: Path | None) -> Code:
    """One script as the text it holds, or what the question says instead of it.

    Read through the workspace, which is the only place that may refuse a path: a script outside the
    workspace is a file this build will not open, and a question that showed nothing would be asking
    somebody to allow a program nobody can read.
    """
    spelled = path if cwd is None else str(cwd / path)
    try:
        where = env.resolve(spelled, AccessMode.READ)
    except PathRefused:
        return Code(says=f"its script {path} is not one this workspace may read, so its code is not here")
    try:
        return Code(text=where.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return Code(says=f"its script {path} could not be read")


async def agreed(env: Env, argv: list[str], *, where: str = "", code: Code | None = None) -> Refused | tuple[str, ...]:
    """Ask whatever this argv needs a person's yes for, and answer the programs a yes may be kept for.

    One question for a command the denylist names — asked program by program, because that is what a yes
    is remembered by — and one question for a command this build cannot answer for, which is asked every
    time because nothing about it is certain enough to keep. When both hold there is one question and no
    memory: a yes to a program whose name is a guess is not a yes to that program.

    ``where`` is what the call is doing it in — ``"on a terminal"`` — because the same argv is a different
    question when it is about to own a screen. ``code`` is the text that is about to run, when the caller
    could find it: a tree's node knows its own input and its own directory, and a caller that hands
    nothing here still gets the code an interpreter carries on its line.
    """
    verdict = env.workspace.commands.verdict(argv)
    if isinstance(verdict, Allowed):
        return ()
    if isinstance(verdict, NeedsReview) or verdict.reasons:
        about = code if code is not None else code_of(unwrap_command(argv).command)
        answer = await _asked(env, _review_prompt(argv, verdict, where=where, code=about), text=about.text)
        return () if answer.accepted else _refusal(answer, about=f"running {shlex.join(argv)}")
    remembered: list[str] = []
    for program in verdict.programs:
        answer = await _asked(env, _denylist_prompt(program, where=where))
        if not answer.accepted:
            return _refusal(answer, about=f"running {program!r}")
        env.workspace.commands.approve(program)
        remembered.append(program)
    return tuple(remembered)


async def _asked(env: Env, prompt: str, *, text: str = "") -> Agreed:
    """Put one question to a person, at the level a program is, and answer what they said."""
    question = Confirmation(prompt=prompt, severity=Severity.DANGER, executable_code=text)
    return as_agreed(await env.ask(question, at="gate"))


def _denylist_prompt(program: str, *, where: str) -> str:
    """The question about one denylisted program, saying what the call is doing it in."""
    place = f" {where}" if where else ""
    return f"run {program!r}{place}? it is on the command denylist"


def _review_prompt(argv: list[str], verdict: NeedsHuman | NeedsReview, *, where: str, code: Code) -> str:
    """The question about a whole command line, saying what could not be judged and what else is on it."""
    place = f" {where}" if where else ""
    says = "; ".join(_reason_says(reason, code) for reason in verdict.reasons)
    if not isinstance(verdict, NeedsReview):
        says = f"{says} — and {', '.join(map(repr, verdict.programs))} is on the command denylist"
    return f"run {shlex.join(argv)}{place}? {says}"


def _reason_says(reason: Reason, code: Code) -> str:
    """One reason, with what the question says about the code that will run or about not having it.

    An interpreter's sentence no longer says this build does not read the code. Either the code travels in
    the question, and the sentence points at it, or the reason's own quote of it is all there is; and when
    the code is a module or a file this build will not open, the question says that instead of nothing.
    """
    if reason.kind is not Review.INTERPRETER:
        return reason.says()
    said = REVIEW_SAYS[reason.kind].format(about=reason.detail or "this call")
    if code.text:
        return f"{said}, and its code is below"
    return f"{said}, and {code.says}" if code.says else reason.says()


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
