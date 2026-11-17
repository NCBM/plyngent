"""The repository this workspace is in, read through the same gate a command goes through.

Five tools and one subject: which checkout this is, what its state and history are, and what changed. All
five are reads, and the argv is what makes that true rather than a promise: ``--no-optional-locks`` keeps
``status`` from refreshing ``.git/index``, ``core.fsmonitor=false`` keeps it from calling a hook that
would, and ``diff``'s two flags keep ``diff.external`` and a ``.gitattributes`` textconv filter from
turning a read into a program the repository's own configuration chose. ``LC_ALL=C`` rides along in the
child's environment for a different reason: git's sentences are what this build reads, and a locale must
not change them.

Two decisions shape the module more than the argv does:

* **The declaration is a command's.** Reading a repository starts a program, so it is judged the way any
  program is — by the command policy, with a person asked when the policy says so (§0.8, E18). That is why
  there is one engine and one gate behind both this module and
  :mod:`~plyngent.kernel.tools.process.execution`, and why every call here has to say what it is for.
* **The surface does not move with the repository.** One set of names and one set of schemas, whatever the
  workspace happens to be; what differs per backend is only an answer (§0.8, E23). ``staged`` is therefore
  a *question* — "the change the next commit would record" — rather than a git concept, and ``vcs_kind``
  is the one place that says which thing is answering and what it can answer.

There is no backend object and no detector registry, because one implementation is not an abstraction and
a registry exists to be reached into. The marker file is the probe and one ``rev-parse`` is the
confirmation: a marker that git refuses is a repository that is not there (a missing thing), not a
protocol error. A repository that starts *above* the workspace is answered rather than refused — refusing
it would make this tool stricter than ``execute(["git", "status"])``, which is the one thing a wrapper
must not be (§0.8, E19).
"""

from __future__ import annotations

import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.failure import Failure
from plyngent.kernel.paths import display
from plyngent.kernel.process import Execution
from plyngent.kernel.process import run as run_tree
from plyngent.kernel.toolresult import Effects, Meta, Text, ToolResult
from plyngent.kernel.tools.process.gate import CommandGate
from plyngent.kernel.toolspec import Group, Mutation, Res, ToolArgs, ToolSpec
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from plyngent.kernel.env import Env

#: What announces a repository: git keeps a directory here, a worktree or a submodule a file.
MARKER = ".git"

#: The one option every call carries: without it ``status`` refreshes ``.git/index``, and a read that
#: writes is not a read.
LOCKLESS = ("--no-optional-locks",)

#: The child's locale, because the words git answers with are read back by this module.
LOCALE = {"LC_ALL": "C"}

#: What ``vcs_kind`` says when nothing in or above the workspace is a repository. Not a failure: most
#: directories are not checkouts, and "there is none" is an answer.
NONE = "none here"

#: Appended to a summary when the repository does not start at the workspace root (E19).
ABOVE = " (repository starts above this workspace)"


class KindArgs(ToolArgs, kw_only=True):
    """Ask which version control answers for this workspace, and what it can answer.

    There is nothing to say about *where*: which repository this is, is a fact about the workspace rather
    than about one call, so a ``cwd`` would let one workspace have two answers.
    """


class StatusArgs(ToolArgs, kw_only=True):
    """Ask what the working tree has that the last commit does not, and which branch it is on.

    The answer is git's own short status with the branch header: one line per path, plus the one line that
    says where the checkout is. Nothing is classified here — a listing that grouped "added" apart from
    "modified" would be this build's reading of another program's format, and it would have to be
    rewritten for every backend (§0.8, E23).
    """


class DiffArgs(ToolArgs, kw_only=True):
    """Show what changed, as a unified diff.

    ``staged`` asks the other question: the difference the *next* commit would record rather than the one
    the last commit recorded. It is asked as a question and not as an option, so a repository without an
    index answers it too — with what it does have, and a word about the difference.

    ``path`` is a pathspec, handed to git as it was given (after ``--``) rather than resolved here: a path
    this build resolved would be held to this workspace's denylist, and a wrapper that refuses a path
    ``execute(["git", "diff", "--", path])`` allows is stricter than the program it wraps.
    """

    path: str = ""
    staged: bool = False


class LogArgs(ToolArgs, kw_only=True):
    """Show the most recent commits, newest first.

    ``limit`` is how many, and it is bounded by the schema for the same reason every bound is: a limit of
    a million is a call the model can see is wrong before it makes it.
    """

    limit: Annotated[int, msgspec.Meta(ge=1, le=100)] = 10


class BranchArgs(ToolArgs, kw_only=True):
    """Show the branch this checkout is on, or the commit it is detached at.

    ``vcs_log`` answers what happened; this answers where the next commit would land.
    """


@dataclass(frozen=True, slots=True)
class Repository:
    """Where a repository starts, as git answers it, and whether that is above this workspace."""

    root: Path
    above: bool


class _Asked:
    """One tool call's gate and record: the git commands it ran, and what it cost.

    One object per call rather than per command, because a tool may need two answers (a detached head is a
    second question) and the record of what ran belongs to the call rather than to one of them.
    """

    env: Env
    gate: CommandGate
    started: float
    ran: list[list[str]]
    truncated: bool

    def __init__(self, env: Env, *, started: float) -> None:
        self.env = env
        self.gate = CommandGate(env.workspace, env)
        self.started = started
        self.ran = []
        self.truncated = False

    async def say(self, argv: list[str], *, precheck: bool = False) -> str | ToolResult:
        """Run one git command and hand back what it said, or the result that says there is nothing.

        ``precheck`` is the one confirmation this module makes before it answers anything: the marker file
        said a repository is here, and this is git agreeing. Its non-zero exit is a missing thing rather
        than a protocol error, because that is what it means — the marker was stale.
        """
        try:
            batch = await run_tree(
                [Execution(argv=argv)],
                cwd=self.env.workspace.root,
                env=LOCALE,
                gate=self.gate,
                scope=self.env.scope("vcs-0"),
            )
        except Cancelled:
            # The engine refuses a stopped turn itself; reaching here means the cancel arrived between two
            # of its checks, so nothing has run since the last one it saw.
            return ToolResult.of(Failure.LIMIT, "stopped")
        if batch.stopped is not None:
            stopped = batch.stopped
            if stopped.denial is not None:
                return ToolResult.denied(stopped.denial, stopped.message, hints=stopped.hints)
            return ToolResult.of(stopped.failure, stopped.message, hints=stopped.hints)
        self.ran.append(argv)
        self.truncated = self.truncated or batch.truncated
        ran = batch.ran[0]
        if ran.exit_code != 0:
            detail = ran.stderr.strip() or ran.stdout.strip() or f"exit {ran.exit_code}"
            return ToolResult.of(Failure.MISSING if precheck else Failure.PROTOCOL, f"git failed: {detail}")
        return ran.stdout

    def answer(self, summary: str, text: str) -> ToolResult:
        """This call's result: the summary a reader takes first, then the words git gave it."""
        return ToolResult.ok(
            summary,
            body=Text(value=text),
            effects=Effects(
                external=tuple(f"ran: {shlex.join(argv)}" for argv in self.ran),
                approvals=tuple(self.gate.approvals),
            ),
            meta=Meta(elapsed_ms=int((time.monotonic() - self.started) * 1000), truncated=self.truncated),
        )


async def vcs_kind(env: Env, _args: KindArgs) -> ToolResult:
    """Which version control answers here, and the three things a model needs to know about it."""
    asked = _Asked(env, started=time.monotonic())
    found = _marker(env.workspace.root)
    if found is None:
        where = display(env.workspace.root)
        return asked.answer(NONE, f"no repository in or above {where}: nothing here is version-controlled")
    repo = await _confirm(asked, found)
    if isinstance(repo, ToolResult):
        return repo
    return asked.answer("git", _described(repo))


async def vcs_status(env: Env, _args: StatusArgs) -> ToolResult:
    """What the working tree has that the last commit does not, branch line included."""
    asked = _Asked(env, started=time.monotonic())
    repo = await _repository(asked)
    if isinstance(repo, ToolResult):
        return repo
    said = await asked.say(_git("-c", "core.fsmonitor=false", "status", "--short", "--branch"))
    if isinstance(said, ToolResult):
        return said
    changed = sum(1 for line in said.splitlines() if not line.startswith("##"))
    return asked.answer(_at("clean" if not changed else f"{counted(changed, 'path')} changed", repo), said or "(clean)")


async def vcs_diff(env: Env, args: DiffArgs) -> ToolResult:
    """The difference this call asked for, counted and then shown."""
    asked = _Asked(env, started=time.monotonic())
    repo = await _repository(asked)
    if isinstance(repo, ToolResult):
        return repo
    argv = _git("diff", "--no-ext-diff", "--no-textconv", *(["--cached"] if args.staged else []))
    if args.path.strip():
        argv += ["--", args.path]
    said = await asked.say(argv)
    if isinstance(said, ToolResult):
        return said
    lines = said.splitlines()
    files = sum(1 for line in lines if line.startswith("diff --git "))
    if not files:
        return asked.answer(_at("no diff", repo), said or "(no diff)")
    added = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
    return asked.answer(_at(f"{counted(files, 'file')} changed, +{added} -{removed}", repo), said)


async def vcs_log(env: Env, args: LogArgs) -> ToolResult:
    """The most recent commits, or the answer that there are none."""
    asked = _Asked(env, started=time.monotonic())
    repo = await _repository(asked)
    if isinstance(repo, ToolResult):
        return repo
    said = await asked.say(_git("log", f"-n{args.limit}", "--format=%h %ad %an %s", "--date=short"))
    if isinstance(said, ToolResult):
        # A repository with no commits is not a failure: git calls it one (`log` exits 128), so the
        # question is asked structurally instead — is there a commit anywhere in this repository?
        if await _unborn(asked):
            return asked.answer(
                _at("no commits", repo),
                "no commits: this repository has none on any branch, so there is no history to show",
            )
        return said
    commits = sum(1 for line in said.splitlines() if line.strip())
    return asked.answer(_at(f"{counted(commits, 'commit')}", repo), said)


async def vcs_branch(env: Env, _args: BranchArgs) -> ToolResult:
    """The branch the next commit would land on, or the commit a detached head is sitting at."""
    asked = _Asked(env, started=time.monotonic())
    repo = await _repository(asked)
    if isinstance(repo, ToolResult):
        return repo
    said = await asked.say(_git("branch", "--show-current"))
    if isinstance(said, ToolResult):
        return said
    name = said.strip()
    if name:
        return asked.answer(_at(f"on {name}", repo), name)
    head = await asked.say(_git("rev-parse", "--short", "HEAD"))
    if isinstance(head, ToolResult):
        return asked.answer(_at("no branch", repo), "no branch: this checkout is not on one")
    return asked.answer(_at(f"detached at {head.strip()}", repo), f"detached at {head.strip()}")


async def _repository(asked: _Asked) -> Repository | ToolResult:
    """The repository this workspace is in, or the result that says there is none."""
    found = _marker(asked.env.workspace.root)
    if found is None:
        return ToolResult.of(Failure.MISSING, f"no repository in or above {display(asked.env.workspace.root)}")
    return await _confirm(asked, found)


async def _confirm(asked: _Asked, found: Path) -> Repository | ToolResult:
    """Whether git answers for the repository the marker announced, and where it says it starts."""
    said = await asked.say(_git("rev-parse", "--show-toplevel"), precheck=True)
    if isinstance(said, ToolResult):
        return said
    root = Path(said.strip()) if said.strip() else found
    return Repository(root=root, above=root.resolve() != asked.env.workspace.root)


async def _unborn(asked: _Asked) -> bool:
    """Whether this repository has no commit at all — the one place git's 128 is a fact rather than a fault."""
    said = await asked.say(_git("rev-list", "--count", "--all"))
    return not isinstance(said, ToolResult) and said.strip() == "0"


def _marker(start: Path) -> Path | None:
    """The directory the marker was found in, walking up from ``start`` — or nothing."""
    for here in (start, *start.parents):
        if (here / MARKER).exists():
            return here
    return None


def _git(*args: str) -> list[str]:
    """One git argv: the program, the option every call carries, and the question."""
    return ["git", *LOCKLESS, *args]


def _at(summary: str, repo: Repository) -> str:
    """A summary, with the workspace's one caveat when the repository does not start at its root."""
    return summary + ABOVE if repo.above else summary


def _described(repo: Repository) -> str:
    """What ``vcs_kind`` answers: where the repository starts, and what it can be asked (E23)."""
    return "\n".join(
        (
            f"repository: {repo.root}{ABOVE if repo.above else ''}",
            "answers: status, diff, log, branch",
            "staged (diff): the difference the next commit would record; this backend keeps an index",
        )
    )


KIND = ToolSpec(
    name="vcs_kind",
    group=Group.VCS,
    args=KindArgs,
    run=vcs_kind,
    uses=Res.EXEC | Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

STATUS = ToolSpec(
    name="vcs_status",
    group=Group.VCS,
    args=StatusArgs,
    run=vcs_status,
    uses=Res.EXEC | Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

DIFF = ToolSpec(
    name="vcs_diff",
    group=Group.VCS,
    args=DiffArgs,
    run=vcs_diff,
    uses=Res.EXEC | Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

LOG = ToolSpec(
    name="vcs_log",
    group=Group.VCS,
    args=LogArgs,
    run=vcs_log,
    uses=Res.EXEC | Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

BRANCH = ToolSpec(
    name="vcs_branch",
    group=Group.VCS,
    args=BranchArgs,
    run=vcs_branch,
    uses=Res.EXEC | Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

SPECS: tuple[ToolSpec, ...] = (KIND, STATUS, DIFF, LOG, BRANCH)
