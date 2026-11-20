"""What the vcs group answers that no story can stage.

A story has a workspace and a person in front of it. What it cannot have is a machine with no git, a
pathspec git itself refuses, or a repository that starts above the workspace — and those are the ways the
answer changes without the tool changing. So they are here, next to the two claims that are about the
declaration, not about a call: what the five tools declare, and what they refuse to let a
repository's own configuration do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from plyngent.kernel.cancel import CancelToken, Scope
from plyngent.kernel.catalog import ToolCatalog
from plyngent.kernel.env import Env
from plyngent.kernel.events import Human, NoSink
from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.toolresult import Status
from plyngent.kernel.tools import vcs
from plyngent.kernel.toolspec import Intention, Mutation, Posture, Res
from plyngent.kernel.workspace import Workspace
from plyngent.wire.ask import AskSite, Severity
from tests import repositories

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.access import AccessMode
    from plyngent.kernel.session import TurnCtx

pytestmark = pytest.mark.skipif(repositories.MISSING, reason="this group reads a repository by starting git")


@dataclass(frozen=True)
class _Host:
    """The two fields this group reaches for, without standing up a whole assembly."""

    workspace: Workspace
    human: Human


@dataclass(frozen=True)
class _Session:
    """As much of a session as these tools need: no grants, since nothing here was allowed one.

    ``session_id`` is what a temporary workspace would be made for and the only owner whose scratch a call
    may reach (E152); none of these tests makes one.
    """

    grants: dict[Path, AccessMode] = field(default_factory=dict)
    session_id: str = "s1"


@dataclass(frozen=True)
class _Ctx:
    """The parts of a turn these tools reach for: its host, and an address to cancel under."""

    host: _Host
    cancel: CancelToken
    session: _Session = field(default_factory=_Session)

    def scope(self, *segments: str) -> Scope:
        return self.cancel.at("1", "r0", *segments)

    def site(self, *segments: str) -> AskSite:
        return AskSite(session="s1", turn=1, scope="/".join(("1", "r0", *segments)))


#: Everything a call may reach: these tests are about what a tool does, not about what it declared.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT | Res.TOOLS

#: The tool set a call may look at here: nothing, because none of these tests asks.
_VIEW = ToolCatalog().view(())


def _env(root: Path) -> Env:
    host = _Host(Workspace(root), NoSink())
    return Env(
        cast("TurnCtx", cast("object", _Ctx(host, CancelToken()))), _EVERYTHING, "tool-1", tool="probe", tools=_VIEW
    )


async def test_a_workspace_that_is_not_a_repository_is_answered_rather_than_refused(tmp_path) -> None:
    """``vcs_kind`` says there is none — most directories are not checkouts — and the other four say why
    they have nothing to answer, which is a missing thing, not a refusal (§0.8, E23)."""
    kind = await vcs.vcs_kind(_env(tmp_path), vcs.KindArgs(intention="what is this"))
    status = await vcs.vcs_status(_env(tmp_path), vcs.StatusArgs(intention="what changed"))

    assert (kind.status, kind.summary) == (Status.OK, vcs.NONE), "an answer, not a failure"
    assert "no repository in or above" in kind.body[0].value
    assert (status.status, status.failure) == (Status.FAILED, Failure.MISSING), "and nothing was refused"
    assert status.retry is Retry.FIX_ARGS, "a checkout is a thing the person can make"


async def test_a_repository_above_the_workspace_is_answered_and_says_so(tmp_path) -> None:
    """E19: refusing it would make this tool stricter than the command it wraps, so the summary says it."""
    repositories.make(tmp_path)
    inside = tmp_path / "sub"
    inside.mkdir()

    status = await vcs.vcs_status(_env(inside), vcs.StatusArgs(intention="what changed"))
    kind = await vcs.vcs_kind(_env(inside), vcs.KindArgs(intention="what is this"))

    assert status.status is Status.OK
    assert status.summary.endswith(vcs.ABOVE), status.summary
    assert kind.summary == "git"
    described = kind.body[0].value
    assert "starts above this workspace" in described, "and it is the one place the root is named"
    assert f"repository: {tmp_path}" in described


async def test_a_machine_without_git_answers_a_missing_thing(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The program this group starts is not one this build ships, so its absence is a missing thing."""
    repositories.make(tmp_path)
    nowhere = tmp_path / "bin"
    nowhere.mkdir()
    monkeypatch.setenv("PATH", str(nowhere))

    status = await vcs.vcs_status(_env(tmp_path), vcs.StatusArgs(intention="what changed"))

    assert (status.status, status.failure) == (Status.FAILED, Failure.MISSING)
    assert "git" in status.summary, "and the answer names what was not there"


async def test_a_pathspec_git_refuses_is_the_argument_error_it_is(tmp_path) -> None:
    """A path is handed to git as it was given, so the one who says a pathspec is wrong is git."""
    repositories.make(tmp_path)

    diff = await vcs.vcs_diff(_env(tmp_path), vcs.DiffArgs(path=":(bogus)x", intention="a broken pathspec"))

    assert (diff.status, diff.failure) == (Status.FAILED, Failure.PROTOCOL)
    assert diff.retry is Retry.FIX_ARGS, "the call is what was wrong, not the world"
    assert "pathspec" in diff.summary, diff.summary


async def test_an_unborn_repository_has_no_commits_rather_than_a_failure(tmp_path) -> None:
    """``git log`` exits 128 on a repository with no commits; that is the answer, not a fault (§0.8)."""
    repositories.empty(tmp_path)

    log = await vcs.vcs_log(_env(tmp_path), vcs.LogArgs(intention="what happened here"))

    assert (log.status, log.summary) == (Status.OK, "no commits")
    assert "no commits" in log.body[0].value


async def test_an_external_diff_driver_is_never_run(tmp_path) -> None:
    """A repository's own configuration is the one way a read becomes somebody else's program.

    ``diff.external`` is a command git runs *in place of* showing a diff, and it is set per repository, so
    a model asked to look at what changed could otherwise run anything the repository's author wrote. The
    ``--no-ext-diff`` in every diff argv is what makes that not happen; this fails if it is ever dropped.
    """
    marker = tmp_path / "ran"
    repositories.make(tmp_path)
    repositories.configure(tmp_path, "diff.external", f"sh -c 'touch {marker}'")
    _ = (tmp_path / "readme.txt").write_text("changed\n", encoding="utf-8")

    diff = await vcs.vcs_diff(_env(tmp_path), vcs.DiffArgs(intention="what changed"))

    assert not marker.exists(), "the repository's own program ran on a read"
    assert diff.status is Status.OK
    assert "-hello" in diff.body[0].value, "and the diff is the bytes, not whatever a filter said"


def test_the_five_declarations_are_one_read_of_a_program() -> None:
    """E18 and E23, as one walk: a command's declaration, an answer that does not move with the backend."""
    assert [spec.name for spec in vcs.SPECS] == [
        "vcs_kind",
        "vcs_status",
        "vcs_diff",
        "vcs_log",
        "vcs_branch",
    ], "the names main had, unchanged: a tool name is not a backend's business"

    for spec in vcs.SPECS:
        assert isinstance(spec.args, type), "a builtin tool's arguments are a package, not somebody else's schema"
        asking = spec.asking(spec.args())
        assert spec.uses == Res.EXEC | Res.PATHS, f"{spec.name} starts a program"
        assert spec.mutates is Mutation.OUTSIDE, "what a backend writes on a read is not this build's to promise"
        assert asking.posture is Posture.AUTO, "the command policy is the one that judges a call"
        assert spec.level() is Severity.DANGER, "and a program is the level a program is"
        assert spec.intention is Intention.REQUIRED, "the first group whose reads have to say why"

    assert vcs.DIFF.roles is None, "a pathspec is git's, not a path this workspace resolves (E21)"
    assert vcs.DIFF.schema()["properties"]["path"]["default"] == ""
    assert vcs.DIFF.schema()["properties"]["staged"]["default"] is False
    limit = vcs.LOG.schema()["properties"]["limit"]
    assert (limit["minimum"], limit["maximum"]) == (1, 100), "a bound is declared, not checked in the handler"
