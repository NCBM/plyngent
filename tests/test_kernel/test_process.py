"""Running commands: one tree per call, a non-zero exit is data, and stopping kills the group."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.failure import Failure
from plyngent.kernel.process import (
    DEFAULT_MAX_NODES,
    MAX_DEPTH,
    OUTPUT_BUDGET,
    Batch,
    Execution,
    FromFile,
    FromText,
    Gate,
    Permit,
    Refused,
    run,
    validate,
)
from plyngent.wire.scope import CancelReason, CancelScope

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from plyngent.kernel.cancel import Scope
    from plyngent.kernel.process import Failfast

if os.name != "posix":  # pragma: no cover — process groups and these shells are a POSIX question
    pytest.skip("process groups are a POSIX question", allow_module_level=True)


def _scope() -> Scope:
    """A cancellation address under one turn, so a test can stop what it started."""
    return CancelToken().scope(CancelScope(("1", "r0", "proc-0")))


def _gate(root: Path, seen: list[Execution] | None = None) -> Gate:
    """A gate that allows everything, resolving the paths a node names the way the tool layer does."""

    async def gate(node: Execution, cwd: Path) -> Permit:
        if seen is not None:
            seen.append(node)
        here = (root / node.cwd).resolve() if node.cwd else cwd
        sink = (here / node.pipe).resolve() if isinstance(node.pipe, str) else None
        source = (here / node.stdin.path).resolve() if isinstance(node.stdin, FromFile) else None
        return Permit(cwd=here, sink=sink, source=source)

    return gate


async def _run(
    tree: list[Execution],
    root: Path,
    *,
    gate: Gate | None = None,
    scope: Scope | None = None,
    env: Mapping[str, str] | None = None,
    failfast: Failfast = True,
    extra_timeout_secs: float = 0.0,
    budget: int = OUTPUT_BUDGET,
) -> Batch:
    """The engine, with the gate and the scope every call needs."""
    return await run(
        tree,
        cwd=root,
        env=env,
        failfast=failfast,
        extra_timeout_secs=extra_timeout_secs,
        gate=gate if gate is not None else _gate(root),
        scope=scope if scope is not None else _scope(),
        budget=budget,
    )


async def test_one_command_is_a_tree_with_one_node(tmp_path) -> None:
    batch = await _run([Execution(argv=["sh", "-c", "echo out; echo err >&2; exit 3"])], tmp_path)

    assert batch.nodes == 1
    assert batch.stopped is None, "a non-zero exit is what the program said, not a failure"
    (ran,) = batch.ran
    assert (ran.exit_code, ran.stdout, ran.stderr) == (3, "out\n", "err\n")
    assert ran.failed is True, "it still did not do what it was asked"


async def test_a_command_that_could_not_start_stops_the_call(tmp_path) -> None:
    batch = await _run([Execution(argv=["definitely-not-a-program"])], tmp_path)

    assert batch.ran == ()
    assert batch.stopped is not None
    assert batch.stopped.failure is Failure.MISSING
    assert "definitely-not-a-program" in batch.stopped.message


async def test_the_continuation_follows_the_exit_code(tmp_path) -> None:
    ok = await _run([Execution(argv=["true"], then=[Execution(argv=["echo", "yes"])])], tmp_path)
    bad = await _run([Execution(argv=["false"], else_=[Execution(argv=["echo", "no"])])], tmp_path)
    stopped = await _run([Execution(argv=["false"])], tmp_path)

    assert [ran.stdout for ran in ok.ran] == ["", "yes\n"]
    assert [ran.stdout for ran in bad.ran] == ["", "no\n"]
    assert stopped.halted is True, "a failed command with nothing after it ends the call without failing it"
    assert stopped.stopped is None


async def test_a_continuation_runs_where_its_parent_ran(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    batch = await _run(
        [Execution(argv=["pwd"], cwd="sub", then=[Execution(argv=["pwd"])], else_=[Execution(argv=["pwd"])])],
        tmp_path,
    )

    assert [ran.cwd for ran in batch.ran] == [str(tmp_path / "sub")] * 2, "the child inherits the directory"


async def test_failfast_false_keeps_going_and_still_cannot_swallow_a_missing_program(tmp_path) -> None:
    kept = await _run([Execution(argv=["false"]), Execution(argv=["echo", "next"])], tmp_path, failfast=False)
    stuck = await _run([Execution(argv=["nope"]), Execution(argv=["echo", "next"])], tmp_path, failfast=False)

    assert [ran.argv for ran in kept.ran] == [("false",), ("echo", "next")]
    assert kept.stopped is None
    assert stuck.ran == (), "it never started, and that is not a non-zero exit to be ignored"
    assert stuck.stopped is not None and stuck.stopped.failure is Failure.MISSING


async def test_a_later_node_asks_a_later_question(tmp_path) -> None:
    seen: list[Execution] = []
    await _run(
        [Execution(argv=["true"], then=[Execution(argv=["echo", "a"])]), Execution(argv=["echo", "b"])],
        tmp_path,
        gate=_gate(tmp_path, seen),
    )

    assert [node.argv for node in seen] == [["true"], ["echo", "a"], ["echo", "b"]], "asked once per node"


async def test_a_branch_that_is_not_taken_is_never_asked_about(tmp_path) -> None:
    seen: list[Execution] = []

    await _run([Execution(argv=["false"], then=[Execution(argv=["never"])])], tmp_path, gate=_gate(tmp_path, seen))

    assert [node.argv for node in seen] == [["false"]], "the success branch was not reached, so nobody asked"


async def test_a_refusal_is_the_call_s_answer_and_not_an_exception(tmp_path) -> None:
    seen: list[Execution] = []
    refused = Refused(Failure.DENIED, "not this one")

    async def gate(node: Execution, cwd: Path) -> Permit | Refused:
        seen.append(node)
        return await _gate(tmp_path)(node, cwd) if node.argv == ["true"] else refused

    batch = await _run([Execution(argv=["true"], then=[Execution(argv=["denied"])])], tmp_path, gate=gate)

    assert [node.argv for node in seen] == [["true"], ["denied"]]
    assert batch.stopped == refused
    assert [ran.argv for ran in batch.ran] == [("true",)], "the refusal does not un-run what ran"


async def test_a_pipe_chain_hands_its_output_along(tmp_path) -> None:
    batch = await _run([Execution(argv=["echo", "hello"], pipe=Execution(argv=["tr", "a-z", "A-Z"]))], tmp_path)

    assert [ran.argv[-1] for ran in batch.ran] == ["hello", "A-Z"]
    assert batch.ran[1].stdout == "HELLO\n"
    assert batch.ran[1].fed is True
    assert (batch.ran[1].depth, batch.ran[1].via) == (1, "pipe")


async def test_a_pipe_can_write_to_a_file_instead_of_a_program(tmp_path) -> None:
    batch = await _run([Execution(argv=["echo", "saved"], pipe="out.txt")], tmp_path)

    assert batch.stopped is None
    assert (tmp_path / "out.txt").read_text() == "saved\n"


async def test_which_command_in_the_chain_decides_depends_on_the_shell_reading(tmp_path) -> None:
    chain = Execution(argv=["false"], pipe=Execution(argv=["echo", "last"]))

    simple = await _run([chain], tmp_path, failfast="simple")
    pipefail = await _run([chain], tmp_path, failfast="pipefail")

    assert simple.halted is False, "the last command of the chain worked"
    assert pipefail.halted is True, "a stage of the chain did not"


async def test_input_comes_from_a_string_a_file_or_the_pipe(tmp_path) -> None:
    _ = (tmp_path / "in.txt").write_text("from a file")

    text = await _run([Execution(argv=["cat"], stdin=FromText(text="from a string"))], tmp_path)
    where = await _run([Execution(argv=["cat"], stdin=FromFile(path="in.txt"))], tmp_path)

    assert text.ran[0].stdout == "from a string"
    assert where.ran[0].stdout == "from a file"


async def test_the_environment_is_this_process_then_the_call_then_the_node(tmp_path) -> None:
    batch = await _run(
        [Execution(argv=["sh", "-c", "echo $A $B"], env={"B": "node"})],
        tmp_path,
        env={"A": "call", "B": "call"},
    )

    assert batch.ran[0].stdout == "call node\n"


@pytest.mark.parametrize("name", ["LD_PRELOAD", "ld_preload", "BASH_ENV", "IFS"])
def test_an_environment_that_changes_how_a_program_loads_code_is_refused(name: str) -> None:
    with pytest.raises(ValueError, match="loads code"):
        _ = Execution(argv=["ls"], env={name: "x"})


def test_a_tree_is_refused_before_anything_runs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _ = validate([])
    with pytest.raises(ValueError, match="stdin comes from the pipe"):
        _ = validate([Execution(argv=["a"], pipe=Execution(argv=["b"], stdin=FromText(text="x")))])
    with pytest.raises(ValueError, match="takes no then/else"):
        _ = validate([Execution(argv=["a"], pipe=Execution(argv=["b"], then=[Execution(argv=["c"])]))])
    with pytest.raises(ValueError, match="more than"):
        _ = validate([Execution(argv=["a"])] * (DEFAULT_MAX_NODES + 1))

    nested = Execution(argv=["a"])
    for _ in range(MAX_DEPTH + 1):
        nested = Execution(argv=["a"], then=[nested])
    with pytest.raises(ValueError, match="nest at most"):
        _ = validate([nested])


@pytest.mark.parametrize("mode", [True, "simple", "pipefail", False])
@pytest.mark.parametrize("situation", ["normal", "nonzero", "missing"])
async def test_failfast_is_the_same_rule_in_every_cell_of_the_table(mode: Failfast, situation: str, tmp_path) -> None:
    """Three readings of the switch against the three things a command can do.

    The row that does not depend on the switch is the point: whatever `failfast` says, a program that
    was never there stops the call, because there is no exit code to branch on and the rest of the tree
    would be guessing.
    """
    argv = {"normal": ["true"], "nonzero": ["false"], "missing": ["definitely-not-a-program"]}[situation]

    batch = await _run([Execution(argv=argv)], tmp_path, failfast=mode)

    if situation == "missing":
        assert batch.ran == ()
        assert batch.stopped is not None and batch.stopped.failure is Failure.MISSING
        assert batch.halted is False
        return
    assert (len(batch.ran), batch.stopped) == (1, None), "the command ran, and what it said is the answer"
    assert batch.halted is (situation == "nonzero" and mode is not False)


async def test_the_two_streams_are_reported_the_way_they_were_asked_for(tmp_path) -> None:
    batch = await _run([Execution(argv=["sh", "-c", "echo o; echo e >&2; exit 2"])], tmp_path)

    (ran,) = batch.ran
    assert (ran.exit_code, ran.stdout, ran.stderr, ran.errout) == (2, "o\n", "e\n", "")


async def test_a_command_that_never_answers_is_stopped_and_said_so(tmp_path) -> None:
    batch = await _run([Execution(argv=["sleep", "30"], timeout_secs=0.05)], tmp_path)

    assert batch.ran[0].timed_out is True
    assert batch.stopped is not None and batch.stopped.failure is Failure.TIMEOUT


async def test_a_command_that_ran_out_of_time_still_says_what_it_had_said(tmp_path) -> None:
    """The deadline ends the waiting, not the record: what a program said before it hung is the answer.

    A kill is what ends a command that never finishes, and it takes whatever had not been read yet with
    it — so this has to be read while it arrives rather than afterwards. `sh -c 'echo hi; sleep 30'` is
    the whole case: everything worth knowing is already on the pipe by the time the deadline is reached
    (§2 R6).
    """
    batch = await _run([Execution(argv=["sh", "-c", "echo hi; sleep 30"], timeout_secs=0.3)], tmp_path)

    (ran,) = batch.ran
    assert (ran.timed_out, ran.stdout) == (True, "hi\n")
    assert ran.escaped is False, "nothing outlived the kill here, and the silence of the pipes says so"
    assert batch.stopped is not None and batch.stopped.failure is Failure.TIMEOUT


async def test_a_process_that_left_the_group_is_said_and_this_end_of_the_pipes_is_closed(tmp_path) -> None:
    """E161: `killpg` reaches a group, and a program that called `setsid` is not in it.

    Nothing portable can promise to kill that one, so two things happen instead of a lie. The answer says
    it — `escaped=true`, which is what a reader looking at a `timed_out` command needs to know — and this
    end of the pipes is closed, which releases the fd, ends the wait at once instead of spending the whole
    grace on an EOF that is never coming, and gives whatever is still writing `EPIPE` the next time it
    says anything.
    """
    if shutil.which("setsid") is None:  # pragma: no cover — the case is util-linux's own program
        pytest.skip("setsid is how a process leaves the group this build kills")
    batch = await _run(
        [Execution(argv=["sh", "-c", 'setsid sh -c "sleep 30" & echo out; sleep 30'], timeout_secs=0.3)],
        tmp_path,
    )

    (ran,) = batch.ran
    assert (ran.timed_out, ran.escaped) == (True, True), "the group was killed and something outlived it"
    assert ran.stdout == "out\n", "and what it had said before the deadline is still the answer"
    assert batch.stopped is not None and batch.stopped.failure is Failure.TIMEOUT


async def test_the_call_can_ask_for_longer_once_instead_of_per_command(tmp_path) -> None:
    batch = await _run([Execution(argv=["sleep", "0.3"], timeout_secs=0.05)], tmp_path, extra_timeout_secs=2)

    assert batch.ran[0].timed_out is False, "the extra is added to every node's deadline"


async def test_stopping_the_turn_kills_the_whole_process_group(tmp_path) -> None:
    token = CancelToken()
    scope = token.scope(CancelScope(("1", "r0", "proc-0")))
    pid_file = tmp_path / "child.pid"
    task = asyncio.ensure_future(
        _run(
            [Execution(argv=["sh", "-c", f"sleep 300 & echo $! > {pid_file}; wait"], timeout_secs=60)],
            tmp_path,
            scope=scope,
        )
    )
    pid = await _appeared(pid_file)

    _ = token.fire(CancelScope(("1", "r0", "proc-0")), CancelReason.USER)
    batch = await task

    assert batch.stopped is not None and batch.stopped.failure is Failure.LIMIT
    await _gone(pid)


async def test_a_timeout_kills_the_whole_process_group(tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    batch = await _run(
        [Execution(argv=["sh", "-c", f"sleep 300 & echo $! > {pid_file}; wait"], timeout_secs=0.2)],
        tmp_path,
    )

    assert batch.stopped is not None and batch.stopped.failure is Failure.TIMEOUT
    await _gone(int(pid_file.read_text()))


async def test_output_beyond_one_command_s_cap_is_cut_and_marked(tmp_path) -> None:
    batch = await _run([Execution(argv=["sh", "-c", "yes x | head -c 200000"])], tmp_path)

    assert batch.truncated is True
    assert batch.ran[0].stdout.endswith("...[truncated at 32000 chars]")


async def test_a_runaway_call_is_stopped_rather_than_truncated_forever(tmp_path) -> None:
    batch = await _run([Execution(argv=["sh", "-c", "yes x | head -c 100000"])], tmp_path, budget=1000)

    assert batch.stopped is not None and batch.stopped.failure is Failure.LIMIT
    assert len(batch.ran) == 1, "what ran did run; it is the rest that did not happen"


async def test_the_merge_of_the_two_streams_leaves_one_stream(tmp_path) -> None:
    batch = await _run([Execution(argv=["sh", "-c", "echo o; echo e >&2"], mix_stderr=True)], tmp_path)

    (ran,) = batch.ran
    assert ran.errout == "o\ne\n", "both streams, in the order the child wrote them"
    assert (ran.stdout, ran.stderr) == ("", "")


async def _appeared(path: Path, *, limit: float = 5.0) -> int:
    """Wait for a pid file a child writes, and answer the pid in it."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if path.exists() and (text := path.read_text().strip()):
            return int(text)
        await asyncio.sleep(0.02)
    pytest.fail(f"{path} never appeared")


async def _gone(pid: int, *, limit: float = 5.0) -> None:
    """Wait until *pid* is no longer there, which is what "the group was killed" means."""
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.02)
    pytest.fail(f"process {pid} is still there after the call stopped")
