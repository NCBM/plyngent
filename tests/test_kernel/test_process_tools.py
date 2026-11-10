"""What is left of the command tool's own tests: the claims no story can make.

Everything else is a story now (:mod:`tests.stories.test_programs`). Three things stayed:

* a ``cwd`` that leaves the workspace is refused with the tools that would lift it — a story cannot
  reach a granted directory, because granting one is the thing it would have to do;
* a denylisted program with *nobody* to ask is refused rather than run — the story's CLI always has a
  person on the other end (see the note in ``docs/rewrite-requirements.md`` §0.3);
* the schema shows a tree rather than a hand-written validator, and the two kinds of input are one
  tagged union: both are claims about the declaration, so they read the schema itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from plyngent.kernel.cancel import CancelToken, Scope
from plyngent.kernel.env import Env
from plyngent.kernel.events import Human, NoSink
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import Execution, FromFile, FromText
from plyngent.kernel.toolresult import Status
from plyngent.kernel.tools.process import execution
from plyngent.kernel.toolspec import Res
from plyngent.kernel.workspace import Workspace, WorkspaceConfig
from plyngent.wire.ask import AskSite

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.session import TurnCtx


@dataclass(frozen=True)
class _Host:
    """The two fields this tool reaches for, without standing up a whole assembly."""

    workspace: Workspace
    human: Human


@dataclass(frozen=True)
class _Ctx:
    """The parts of a turn these tools reach for: its host, and an address to cancel under."""

    host: _Host
    cancel: CancelToken

    def scope(self, *segments: str) -> Scope:
        return self.cancel.at("1", "r0", *segments)

    def site(self, *segments: str) -> AskSite:
        return AskSite(session="s1", turn=1, scope="/".join(("1", "r0", *segments)))


#: Everything a call may reach: these tests are about what a tool does, not about what it declared.
_EVERYTHING = Res.PATHS | Res.EXEC | Res.PTY | Res.NET | Res.HUMAN | Res.HISTORY | Res.DOCUMENT | Res.CLIENT


def _env(root: Path, *, human: Human | None = None) -> Env:
    host = _Host(Workspace(root), human if human is not None else NoSink())
    return Env(cast("TurnCtx", cast("object", _Ctx(host, CancelToken()))), _EVERYTHING, "tool-1")


async def test_a_directory_outside_the_workspace_is_refused_with_the_tool_that_lifts_it(tmp_path) -> None:
    result = await execution.execute(
        _env(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["true"])], cwd="../elsewhere")
    )

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED
    assert [hint.tool for hint in result.hints] == ["request_directory_access", "new_temporary_workspace"]


async def test_with_nobody_to_ask_a_denylisted_program_is_refused_rather_than_run(tmp_path) -> None:
    ctx = _env(tmp_path)
    ctx.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"echo"})))

    result = await execution.execute(ctx, execution.ExecuteArgs(steps=[Execution(argv=["echo", "hi"])]))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED


def test_the_schema_shows_the_tree_rather_than_a_hand_written_validator() -> None:
    schema = execution.EXECUTE.schema()

    steps = schema["properties"]["steps"]
    assert steps["type"] == "array"
    assert steps["items"] == {"$ref": "#/$defs/Execution"}
    node = schema["$defs"]["Execution"]["properties"]
    assert set(node) >= {"argv", "cwd", "env", "stdin", "timeout_secs", "failfast", "then", "else", "pipe"}
    assert node["failfast"]["anyOf"], "three readings of one switch, all visible"
    assert "else" in node, "a Python keyword is renamed on the wire"
    assert schema["required"] == ["steps"], "everything else has a default"


def test_an_input_is_one_of_the_two_kinds_it_can_be() -> None:
    parsed = execution.EXECUTE.decode(
        b'{"intention": "read two files", "steps": ['
        b'{"argv": ["cat"], "stdin": {"type": "text", "text": "x"}},'
        b'{"argv": ["cat"], "stdin": {"type": "file", "path": "x"}}'
        b"]}"
    )

    assert isinstance(parsed.steps[0].stdin, FromText)
    assert isinstance(parsed.steps[1].stdin, FromFile), "the tag is what says which one it is"
