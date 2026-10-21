"""Aggregated soft confirm: one prompt for a round's mutating calls."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import (
    BatchConfirmDecision,
    ToolRegistry,
    ToolTag,
    tool,
)
from plyngent.agent.loop import _execute_tool_calls
from plyngent.lmproto.openai_compatible.model import AssistantFunctionTool, AssistantFunctionToolCall
from plyngent.tools.context import SessionState, bind_tool_context
from plyngent.tools.grants import has_grant

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plyngent.agent import ToolConfirmRequest
    from plyngent.lmproto.openai_compatible.model import AnyChatMessage


def _danger(name: str, _args: Mapping[str, object]) -> str | None:
    return f"mutate {name}"


async def test_loop_round_prompts_once_before_running_its_calls() -> None:
    """The loop aggregates a round's confirms and applies one answer to each call."""
    batch_prompts: list[int] = []
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append(len(requests))
        return BatchConfirmDecision(allowed=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry([stamp], danger=_danger, on_confirm=on_single, on_confirm_batch=on_batch)
    tool_calls = [
        AssistantFunctionToolCall(id=f"c{index}", function=AssistantFunctionTool(name="stamp", arguments=arguments))
        for index, arguments in enumerate(['{"path": "a"}', '{"path": "b"}'])
    ]
    messages: list[AnyChatMessage] = []

    events = [event async for event in _execute_tool_calls(reg, tool_calls, messages)]

    assert batch_prompts == [2]
    assert single_prompts == []
    assert [getattr(message, "content", None) for message in messages] == ["stamped a", "stamped b"]
    assert len(events) == len(tool_calls) * 2  # call event + result event per call


async def test_round_asks_once_for_many_mutating_calls() -> None:
    """Three calls in one round share one prompt instead of three."""
    ran: list[str] = []
    batch_prompts: list[list[ToolConfirmRequest]] = []
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.YOLO, register=False)
    async def stamp(path: str) -> str:
        ran.append(path)
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append(list(requests))
        return BatchConfirmDecision(allowed=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry([stamp], danger=_danger, on_confirm=on_single, on_confirm_batch=on_batch)
    calls = [("stamp", '{"path": "a"}'), ("stamp", '{"path": "b"}'), ("stamp", '{"path": "c"}')]
    decisions = await reg.prepare_batch_confirm(calls)

    assert all(decision is not None and decision.allowed for decision in decisions)
    assert [request.args["path"] for request in batch_prompts[0]] == ["a", "b", "c"]
    assert single_prompts == []

    results = [
        await reg.execute(name, arguments, decision=decisions[index]) for index, (name, arguments) in enumerate(calls)
    ]
    assert results == ["stamped a", "stamped b", "stamped c"]
    assert ran == ["a", "b", "c"]
    assert single_prompts == []


async def test_yolo_calls_are_left_out_of_the_prompt() -> None:
    """Under YOLO mode a YOLO-tagged call runs unprompted, batch prompt included."""
    batch_prompts: list[list[str]] = []
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.YOLO, register=False)
    async def yolo_ok(path: str) -> str:
        return f"stamped {path}"

    @tool(tags=ToolTag.LOCAL, register=False)
    async def always_ask(path: str) -> str:
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append([request.name for request in requests])
        return BatchConfirmDecision(allowed=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry(
        [yolo_ok, always_ask],
        danger=_danger,
        on_confirm=on_single,
        on_confirm_batch=on_batch,
        yolo=True,
    )
    calls = [("yolo_ok", '{"path": "a"}'), ("always_ask", '{"path": "b"}'), ("always_ask", '{"path": "c"}')]
    decisions = await reg.prepare_batch_confirm(calls)

    assert decisions[0] is None
    assert batch_prompts == [["always_ask", "always_ask"]]
    results = [
        await reg.execute(name, arguments, decision=decisions[index]) for index, (name, arguments) in enumerate(calls)
    ]
    assert results == ["stamped a", "stamped b", "stamped c"]
    assert single_prompts == []


async def test_single_request_keeps_the_one_call_prompt() -> None:
    """One mutating call stays the plain per-call confirm."""
    batch_prompts: list[int] = []
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append(len(requests))
        return BatchConfirmDecision(allowed=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry([stamp], danger=_danger, on_confirm=on_single, on_confirm_batch=on_batch)
    assert await reg.prepare_batch_confirm([("stamp", '{"path": "a"}')]) == [None]
    assert batch_prompts == []
    assert await reg.execute("stamp", '{"path": "a"}') == "stamped a"
    assert single_prompts == ["stamp"]


async def test_calls_that_cannot_run_are_left_out_of_the_prompt() -> None:
    """Missing arguments, bad JSON and unknown tools never reach the user."""
    batch_prompts: list[list[str]] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str, *, force: bool = False) -> str:
        return f"stamped {path} ({force})"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append([str(request.args.get("path")) for request in requests])
        return BatchConfirmDecision(allowed=True)

    reg = ToolRegistry([stamp], danger=_danger, on_confirm_batch=on_batch)
    decisions = await reg.prepare_batch_confirm(
        [
            ("stamp", "{}"),  # missing ``path``
            ("stamp", "not json"),
            ("nope", "{}"),
            ("stamp", '{"path": "a"}'),
            ("stamp", '{"path": "b", "force": true}'),
        ]
    )

    assert decisions[:3] == [None, None, None]
    assert all(decision is not None for decision in decisions[3:])
    assert batch_prompts == [["a", "b"]]
    missing = await reg.execute("stamp", "{}")
    assert missing.startswith("error: `stamp` is missing required argument")


async def test_batch_deny_reaches_every_call_with_one_comment() -> None:
    """A deny stops the whole batch and hands the same comment to each call."""
    ran: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        ran.append(path)
        return f"stamped {path}"

    def on_batch(_requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        return BatchConfirmDecision(allowed=False, comment="not these two")

    reg = ToolRegistry([stamp], danger=_danger, on_confirm_batch=on_batch)
    calls = [("stamp", '{"path": "a"}'), ("stamp", '{"path": "b"}')]
    decisions = await reg.prepare_batch_confirm(calls)

    results = [
        await reg.execute(name, arguments, decision=decisions[index]) for index, (name, arguments) in enumerate(calls)
    ]
    assert all(result.startswith("error: tool 'stamp' denied by user confirm") for result in results)
    assert all("user comment: not these two" in result for result in results)
    assert ran == []


async def test_each_round_asks_again() -> None:
    """An answer covers one round: the next round's calls are asked again."""
    batch_prompts: list[int] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append(len(requests))
        return BatchConfirmDecision(allowed=True)

    reg = ToolRegistry([stamp], danger=_danger, on_confirm_batch=on_batch)
    for first, second in (("a", "b"), ("c", "d")):
        calls = [("stamp", f'{{"path": "{first}"}}'), ("stamp", f'{{"path": "{second}"}}')]
        decisions = await reg.prepare_batch_confirm(calls)
        for index, (name, arguments) in enumerate(calls):
            await reg.execute(name, arguments, decision=decisions[index])
    assert batch_prompts == [2, 2]


async def test_manual_hands_the_round_back_to_the_per_call_prompt() -> None:
    """``each`` answers nothing for the batch: every call is asked on its own."""
    batch_prompts: list[int] = []
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        return f"stamped {path}"

    def on_batch(requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        batch_prompts.append(len(requests))
        return BatchConfirmDecision(allowed=False, each=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry([stamp], danger=_danger, on_confirm=on_single, on_confirm_batch=on_batch)
    calls = [("stamp", '{"path": "a"}'), ("stamp", '{"path": "b"}')]
    decisions = await reg.prepare_batch_confirm(calls)

    assert decisions == [None, None]
    assert [await reg.execute(name, arguments) for name, arguments in calls] == ["stamped a", "stamped b"]
    assert batch_prompts == [2]
    assert single_prompts == ["stamp", "stamp"]


async def test_without_a_batch_hook_every_call_prompts_on_its_own() -> None:
    """A host that installs no batch hook keeps the per-call prompts."""
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def stamp(path: str) -> str:
        return f"stamped {path}"

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    reg = ToolRegistry([stamp], danger=_danger, on_confirm=on_single)
    calls = [("stamp", '{"path": "a"}'), ("stamp", '{"path": "b"}')]
    assert await reg.prepare_batch_confirm(calls) == [None, None]
    assert [await reg.execute(name, arguments) for name, arguments in calls] == ["stamped a", "stamped b"]
    assert single_prompts == ["stamp", "stamp"]


async def test_batch_allow_records_the_trustable_grant() -> None:
    """A batch allow still carries the trust a single allow would record."""
    single_prompts: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.TRUSTABLE, register=False)
    async def trust_me(path: str) -> str:
        return f"done {path}"

    def on_batch(_requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        return BatchConfirmDecision(allowed=True)

    async def on_single(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        single_prompts.append(name)
        return True

    session = SessionState(session_id=1)
    reg = ToolRegistry(
        [trust_me],
        danger=_danger,
        on_confirm=on_single,
        on_confirm_batch=on_batch,
        auto_bind_state=True,
        session_state=session,
    )
    with bind_tool_context(session=session):
        calls = [("trust_me", '{"path": "a"}'), ("trust_me", '{"path": "b"}')]
        decisions = await reg.prepare_batch_confirm(calls)
        for path, decision in zip(("a", "b"), decisions, strict=True):
            assert await reg.execute("trust_me", f'{{"path": "{path}"}}', decision=decision) == f"done {path}"
        assert has_grant(session, "trust_me")
        # The TRUSTABLE grant, not the batch answer, keeps later calls unchecked.
        assert await reg.execute("trust_me", '{"path": "c"}') == "done c"
    assert single_prompts == []
