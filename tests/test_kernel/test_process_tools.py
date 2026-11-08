"""The ``execute`` tool: one tree per call, a per-node question, and a record instead of a paragraph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, override

from plyngent.kernel.cancel import CancelToken, Scope
from plyngent.kernel.events import Human, NoSink
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import Execution, FromFile, FromText
from plyngent.kernel.render import render
from plyngent.kernel.toolresult import ProcessControl, Status, ToolResult
from plyngent.kernel.tools.process import execution
from plyngent.kernel.toolspec import Intention, Posture, Res
from plyngent.kernel.workspace import Workspace, WorkspaceConfig
from plyngent.wire.ask import Agreed, AskSite, Picked, Written
from plyngent.wire.event import Event

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.kernel.session import TurnCtx


class _Answering(Human):
    """A human who always says the same thing, and remembers what they were asked."""

    def __init__(self, *, yes: bool) -> None:
        self.yes = yes
        self.asked: list[str] = []

    @override
    def emit(self, event: Event) -> None:
        return None

    @override
    async def confirm(self, *, prompt: str, default: bool | None = None, site: AskSite | None = None) -> Agreed:
        _ = (default, site)
        self.asked.append(prompt)
        return Agreed(accepted=self.yes)

    @override
    async def choose(
        self,
        *,
        prompt: str,
        options: tuple[str, ...],
        default: str | None = None,
        site: AskSite | None = None,
    ) -> Picked:
        _ = (default, site)
        self.asked.append(prompt)
        return Picked(value=options[0] if options else None)

    @override
    async def text(self, *, prompt: str, default: str | None = None, site: AskSite | None = None) -> Written:
        _ = (default, site)
        self.asked.append(prompt)
        return Written(text="")

    @override
    async def secret(self, *, prompt: str, site: AskSite | None = None) -> Written:
        _ = site
        self.asked.append(prompt)
        return Written()


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


def _ctx(root: Path, *, human: Human | None = None) -> TurnCtx:
    host = _Host(Workspace(root), human if human is not None else NoSink())
    return cast("TurnCtx", cast("object", _Ctx(host, CancelToken())))


def _nodes(result: ToolResult) -> list[ProcessControl]:
    """The command records in a result, in the order they ran."""
    parts = [part for part in result.body if isinstance(part, ProcessControl)]
    assert len(parts) == len(result.body), "an execution reports nothing but commands"
    return parts


async def test_one_command_is_reported_as_what_it_said(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["sh", "-c", "echo hi; exit 4"])])
    )

    assert result.status is Status.OK, "a non-zero exit is the program's answer, not a failed call"
    assert result.summary == "1 of 1 commands ran"
    (ran,) = _nodes(result)
    assert (ran.argv, ran.exit_code, ran.stdout) == (("sh", "-c", "echo hi; exit 4"), 4, "hi\n")
    assert ran.cwd == ".", "the directory is shown the short way"
    assert result.meta.chars > 0, "the count is of what was projected, and it was"


async def test_a_program_that_was_not_there_fails_the_call_before_anything_ran(tmp_path) -> None:
    result = await execution.execute(_ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["no-such-prog"])]))

    assert result.status is Status.FAILED
    assert result.failure is Failure.MISSING
    assert result.body == ()
    assert "no-such-prog" in result.summary


async def test_a_directory_outside_the_workspace_is_refused_with_the_tool_that_lifts_it(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["true"])], cwd="../elsewhere")
    )

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED
    assert [hint.tool for hint in result.hints] == ["request_directory_access", "new_temporary_workspace"]


async def test_a_denylisted_program_is_put_to_the_human_once_per_program(tmp_path) -> None:
    asked = _Answering(yes=True)
    ctx = _ctx(tmp_path, human=asked)
    ctx.host.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"echo"})))
    args = execution.ExecuteArgs(steps=[Execution(argv=["echo", "hi"]), Execution(argv=["echo", "again"])])

    result = await execution.execute(ctx, args)
    again = await execution.execute(ctx, args)

    assert result.status is Status.OK
    assert len(asked.asked) == 1, "one program, one question — the answer is remembered by program"
    assert result.effects.approvals == ("echo",)
    assert again.effects.approvals == (), "the second call asks nothing: the policy already said yes"


async def test_a_denied_program_stops_the_call_and_nothing_runs(tmp_path) -> None:
    refused = _Answering(yes=False)
    ctx = _ctx(tmp_path, human=refused)
    ctx.host.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"echo"})))

    result = await execution.execute(ctx, execution.ExecuteArgs(steps=[Execution(argv=["echo", "hi"])]))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED
    assert result.effects.external == ()
    assert len(refused.asked) == 1


async def test_with_nobody_to_ask_a_denylisted_program_is_refused_rather_than_run(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    ctx.host.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"echo"})))

    result = await execution.execute(ctx, execution.ExecuteArgs(steps=[Execution(argv=["echo", "hi"])]))

    assert result.status is Status.REFUSED
    assert result.failure is Failure.DENIED


async def test_a_later_node_is_asked_about_only_when_it_is_reached(tmp_path) -> None:
    asked = _Answering(yes=False)
    ctx = _ctx(tmp_path, human=asked)
    ctx.host.workspace.configure(WorkspaceConfig(command_denylist=frozenset({"echo"})))

    result = await execution.execute(
        ctx,
        execution.ExecuteArgs(
            steps=[
                Execution(argv=["false"], then=[Execution(argv=["echo", "skipped"])]),
                Execution(argv=["true"]),
            ],
            failfast=False,
        ),
    )

    assert [ran.argv[0] for ran in _nodes(result)] == ["false", "true"]
    assert asked.asked == [], "the then-branch was never reached, so nobody was asked about it"


async def test_a_failure_with_nothing_after_it_ends_the_call_without_failing_it(tmp_path) -> None:
    result = await execution.execute(_ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["false"])]))

    assert result.status is Status.OK, "the call did what it was asked; the command is what failed"
    assert result.summary == "1 of 1 commands ran"
    assert _nodes(result)[0].exit_code == 1


async def test_failfast_false_follows_the_failure_and_runs_the_next_node(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path),
        execution.ExecuteArgs(
            steps=[Execution(argv=["false"]), Execution(argv=["echo", "next"])],
            failfast=False,
        ),
    )

    assert [ran.argv[0] for ran in _nodes(result)] == ["false", "echo"]
    assert result.summary == "2 of 2 commands ran"


async def test_what_ran_before_a_stop_is_partial_and_not_a_failure(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path),
        execution.ExecuteArgs(
            steps=[Execution(argv=["echo", "first"]), Execution(argv=["no-such-prog"])],
        ),
    )

    assert result.status is Status.PARTIAL, "one command did run, and it stays run"
    assert result.effects.external == ("ran: echo first",)
    assert "no-such-prog" in result.summary
    assert [ran.argv[0] for ran in _nodes(result)] == ["echo"]


async def test_a_deadline_is_reported_with_the_argument_that_would_have_let_it_finish(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["sleep", "5"], timeout_secs=0.05)])
    )

    assert result.status is Status.PARTIAL, "it started, so it is in the record, and a partial is never retried"
    assert "no answer within" in result.summary
    assert _nodes(result)[0].timed_out is True
    assert dict(result.hints[0].given) == {"extra_timeout_secs": "60"}


async def test_a_redirect_that_leaves_the_workspace_is_refused_before_it_runs(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["echo", "x"], pipe="../../out.txt")])
    )

    assert result.status is Status.REFUSED
    assert result.failure is Failure.BLOCKED
    assert result.body == ()


async def test_a_string_on_stdin_is_the_input(tmp_path) -> None:
    result = await execution.execute(
        _ctx(tmp_path), execution.ExecuteArgs(steps=[Execution(argv=["cat"], stdin=FromText(text="typed"))])
    )

    assert _nodes(result)[0].stdout == "typed"


async def test_a_tree_that_could_not_run_is_an_argument_error_and_asks_nobody(tmp_path) -> None:
    result = await execution.execute(_ctx(tmp_path), execution.ExecuteArgs(steps=[]))

    assert result.status is Status.FAILED
    assert result.failure is Failure.ARGS
    assert "must not be empty" in result.summary


async def test_what_the_model_reads_has_the_fields_the_old_tool_printed(tmp_path) -> None:
    """The shape `main`'s `run_argv` printed, so a reader (and a person) does not have to relearn it."""
    ctx = _ctx(tmp_path)

    result = await execution.execute(
        ctx, execution.ExecuteArgs(steps=[Execution(argv=["sh", "-c", "echo out; exit 2"], cwd=".", mix_stderr=False)])
    )

    lines = render(result).splitlines()
    assert result.summary in lines[0]
    assert lines[1:6] == ["exit_code=2", "timed_out=false", "cwd=.", "cmd=sh -c 'echo out; exit 2'", "--- stdout ---"]
    assert lines[6] == "out"


def test_the_declaration_says_it_runs_programs_so_it_has_to_be_told_why() -> None:
    spec = execution.EXECUTE

    assert spec.uses & Res.EXEC
    assert spec.intention is Intention.REQUIRED
    assert spec.approval(execution.ExecuteArgs(steps=[Execution(argv=["ls"])]))[0] is Posture.AUTO, (
        "whether this argv is dangerous is the command policy's answer, not the declaration's"
    )


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
