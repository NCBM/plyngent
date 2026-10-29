"""The ``[approval]`` line: a confirmed call says how it got through."""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import (
    Approval,
    BatchConfirmDecision,
    ToolRegistry,
    ToolTag,
    format_approval,
    most_specific_approval,
    tool,
)
from plyngent.agent.budget import cap_tool_message
from plyngent.lmproto.openai_compatible.model import ToolChatMessage
from plyngent.tools.context import SessionState, bind_tool_context, record_approval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plyngent.agent import ToolConfirmRequest


def _danger(name: str, _args: Mapping[str, object]) -> str | None:
    return f"mutate {name}"


def _allow(_name: str, _args: Mapping[str, object], _reason: str) -> bool:
    return True


@tool(register=False)
async def quiet() -> str:
    return "quiet"


@tool(register=False)
async def mutate() -> str:
    return "ran"


async def test_a_call_that_needs_no_confirmation_carries_no_line() -> None:
    """The common case stays byte-for-byte what the tool returned."""

    def danger(_name: str, _args: Mapping[str, object]) -> str | None:
        return None

    def on_confirm(_name: str, _args: Mapping[str, object], _reason: str) -> bool:
        msg = "no prompt expected"
        raise AssertionError(msg)

    reg = ToolRegistry([quiet], danger=danger, on_confirm=on_confirm)
    assert await reg.execute("quiet", "{}") == "quiet"


async def test_a_single_prompt_leads_the_result() -> None:
    reg = ToolRegistry([mutate], danger=_danger, on_confirm=_allow)
    assert await reg.execute("mutate", "{}") == "[approval] user (once)\nran"


async def test_yolo_mode_says_it_dropped_the_prompt() -> None:
    @tool(tags=ToolTag.LOCAL | ToolTag.YOLO, register=False)
    async def yolo_ok() -> str:
        return "y"

    reg = ToolRegistry([yolo_ok], danger=_danger, on_confirm=_allow, yolo=True)
    assert await reg.execute("yolo_ok", "{}") == "[approval] yolo (auto)\ny"


async def test_a_batch_answer_covers_its_round() -> None:
    def on_batch(_requests: Sequence[ToolConfirmRequest]) -> BatchConfirmDecision:
        return BatchConfirmDecision(allowed=True)

    reg = ToolRegistry([mutate], danger=_danger, on_confirm_batch=on_batch)
    calls = [("mutate", "{}"), ("mutate", "{}")]
    decisions = await reg.prepare_batch_confirm(calls)
    results = [
        await reg.execute(name, arguments, decision=decisions[index]) for index, (name, arguments) in enumerate(calls)
    ]
    assert results == ["[approval] user (round)\nran", "[approval] user (round)\nran"]


async def test_a_session_grant_says_nobody_was_asked_this_time() -> None:
    @tool(tags=ToolTag.LOCAL | ToolTag.TRUSTABLE, register=False)
    async def trust_me() -> str:
        return "ok"

    prompts: list[str] = []

    def on_confirm(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        prompts.append(name)
        return True

    session = SessionState(session_id=1)
    reg = ToolRegistry([trust_me], danger=_danger, on_confirm=on_confirm, session_state=session)
    with bind_tool_context(session=session):
        assert await reg.execute("trust_me", "{}") == "[approval] user (once)\nok"
        assert await reg.execute("trust_me", "{}") == "[approval] grant (session)\nok"
    assert prompts == ["trust_me"]


async def test_a_denied_call_carries_no_line() -> None:
    reg = ToolRegistry([mutate], danger=_danger, on_confirm=lambda *_args: False)
    out = await reg.execute("mutate", "{}")
    assert out.startswith("error: ")
    assert "[approval]" not in out


async def test_a_failed_call_carries_no_line() -> None:
    """An approved call that failed is an error first; the answer explains nothing."""

    @tool(register=False)
    async def boom() -> str:
        return "error: boom"

    reg = ToolRegistry([boom], danger=_danger, on_confirm=_allow)
    assert await reg.execute("boom", "{}") == "error: boom"


async def test_a_handler_record_leads_the_result() -> None:
    @tool(register=False)
    async def policy_tool() -> str:
        record_approval(Approval("user", "process"))
        return "ran"

    reg = ToolRegistry([policy_tool])
    assert await reg.execute("policy_tool", "{}") == "[approval] user (process)\nran"


async def test_the_closest_answer_wins() -> None:
    """YOLO let the call run, but a human answered a policy prompt inside it."""

    @tool(tags=ToolTag.LOCAL | ToolTag.YOLO, register=False)
    async def yolo_policy() -> str:
        record_approval(Approval("user", "process"))
        return "ran"

    reg = ToolRegistry([yolo_policy], danger=_danger, on_confirm=_allow, yolo=True)
    assert await reg.execute("yolo_policy", "{}") == "[approval] user (process)\nran"


async def test_the_line_survives_the_wire_cap() -> None:
    approved = f"{format_approval(Approval('user', 'once'))}\n{'x' * 500}"
    capped = cap_tool_message(ToolChatMessage(content=approved, tool_call_id="call-1"), 200)
    assert capped.content.startswith("[approval] user (once)\n")
    assert "document_id=call-1" in capped.content


def test_the_most_specific_record_wins() -> None:
    records = [Approval("yolo", "auto"), Approval("grant", "session"), Approval("user", "once")]
    assert most_specific_approval(records) == Approval("user", "once")
    assert most_specific_approval([]) is None
