"""Rollback of a failed turn keeps the tool work that already ran.

``committed_prefix_end`` commits a batch only once every call has its result, so
``rollback_tail`` keeps the executed part of a batch that was cut short: the
retry then continues from real results instead of running those tools again.
"""

from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.chat import rollback_tail
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    ToolChatMessage,
    UserChatMessage,
)


def _call(call_id: str, name: str = "some_tool") -> AssistantFunctionToolCall:
    return AssistantFunctionToolCall(id=call_id, function=AssistantFunctionTool(name=name, arguments="{}"))


def test_rollback_tail_keeps_the_executed_call_of_a_cut_short_batch() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="do both"),
        AssistantChatMessage(content="first, then the other", tool_calls=[_call("call_a"), _call("call_b")]),
        ToolChatMessage(tool_call_id="call_a", content="a ran"),
    ]

    tail = rollback_tail(messages, 1)

    assert len(tail) == 2
    record, result = tail
    assert isinstance(record, AssistantChatMessage)
    # The assistant row survives only as the carrier of the pairing.
    assert record.content is UNSET
    assert record.reasoning_content is UNSET
    calls = record.tool_calls
    assert calls is not UNSET
    assert [call.id for call in calls] == ["call_a"]
    assert isinstance(result, ToolChatMessage)
    assert result.content == "a ran"


def test_rollback_tail_drops_a_batch_that_reported_nothing() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="go"),
        AssistantChatMessage(content="about to call", tool_calls=[_call("call_a")]),
    ]

    assert rollback_tail(messages, 1) == []


def test_rollback_tail_drops_unfinished_text_and_keeps_developer_rows() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="go"),
        AssistantChatMessage(content=UNSET, tool_calls=[_call("call_a")]),
        ToolChatMessage(tool_call_id="call_a", content="a ran"),
        DeveloperChatMessage(content="checkpoint"),
        AssistantChatMessage(content="half-written answer"),
    ]

    tail = rollback_tail(messages, 1)

    assert [type(message) for message in tail] == [AssistantChatMessage, ToolChatMessage, DeveloperChatMessage]


def test_rollback_tail_pairs_a_repeated_call_id_once() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="go"),
        AssistantChatMessage(content=UNSET, tool_calls=[_call("call_a"), _call("call_a")]),
        ToolChatMessage(tool_call_id="call_a", content="only once"),
    ]

    tail = rollback_tail(messages, 1)

    assert len(tail) == 2
    record, result = tail
    assert isinstance(record, AssistantChatMessage)
    calls = record.tool_calls
    assert calls is not UNSET
    assert [call.id for call in calls] == ["call_a"]
    assert isinstance(result, ToolChatMessage)
    assert result.content == "only once"


def test_rollback_tail_ignores_rows_before_start() -> None:
    messages: list[AnyChatMessage] = [
        UserChatMessage(content="go"),
        AssistantChatMessage(content=UNSET, tool_calls=[_call("call_a")]),
        ToolChatMessage(tool_call_id="call_a", content="a ran"),
    ]

    assert rollback_tail(messages, len(messages)) == []
