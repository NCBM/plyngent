"""Tests for committed-prefix rollback preserving synthetic todo nag pairs.

Covers committed_prefix_end: after a failed turn, rollback must keep committed
tool batches (including synthetic todo nag pairs) so retry does not re-execute
side effects or duplicate nags.
"""

from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.chat import committed_prefix_end
from plyngent.agent.todo_nag import (
    _append_synthetic_todo_list,
    is_synthetic_todo_nag_call_id,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)


def test_synthetic_pair_survives_rollback() -> None:
    """The synthetic pair survives committed_prefix_end + deletion."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    _append_synthetic_todo_list(messages, "open task: fix bug")

    user_index = 1
    end = committed_prefix_end(messages, user_index)
    del messages[end:]

    # Pair should still be in messages
    assert len(messages) == 4
    assert isinstance(messages[2], AssistantChatMessage)
    assert isinstance(messages[3], ToolChatMessage)


def test_synthetic_pair_survives_rollback_with_subsequent_committed_content() -> None:
    """The pair survives when there's also committed content after it."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    _append_synthetic_todo_list(messages, "open task: fix bug")

    # Add committed tool batch (real tool call + result)
    real_call_id = "call_real_1"
    messages.append(
        AssistantChatMessage(
            content=UNSET,
            tool_calls=[
                AssistantFunctionToolCall(
                    id=real_call_id,
                    function=AssistantFunctionTool(
                        name="some_tool",
                        arguments="{}",
                    ),
                )
            ],
        )
    )
    messages.append(ToolChatMessage(tool_call_id=real_call_id, content="done"))

    # Add uncommitted content that should be removed
    messages.append(AssistantChatMessage(content="uncommitted response"))

    user_index = 1
    end = committed_prefix_end(messages, user_index)
    # end should be 6 (user+1=2, synth_pair=2->4, real_tool=4->6, then stop at 6)
    assert end == 6, f"Expected end=6, got {end}"
    del messages[end:]

    # Both the synthetic pair and the real tool batch survive.
    assert len(messages) == 6


def test_multiple_rollbacks_preserve_single_pair() -> None:
    """Repeated rollbacks don't accumulate duplicate synthetic pairs."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    _append_synthetic_todo_list(messages, "task 1")

    user_index = 1

    # Simulate first failure: roll back uncommitted content
    end = committed_prefix_end(messages, user_index)
    del messages[end:]

    # Simulate second failure: roll back again
    end = committed_prefix_end(messages, user_index)
    del messages[end:]

    # Count pairs
    count = sum(
        1
        for i, m in enumerate(messages)
        if isinstance(m, AssistantChatMessage)
        and m.tool_calls is not UNSET
        and m.tool_calls
        and all(isinstance(c, AssistantFunctionToolCall) and is_synthetic_todo_nag_call_id(c.id) for c in m.tool_calls)
        and i + 1 < len(messages)
        and isinstance(messages[i + 1], ToolChatMessage)
    )
    assert count == 1, f"Expected 1 synthetic pair, got {count}"


def test_rollback_keeps_real_tool_batch_not_matching() -> None:
    """Real tool calls are committed prefix, not treated as synthetic nag."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
        AssistantChatMessage(
            content=UNSET,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_real",
                    function=AssistantFunctionTool(name="tool", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(tool_call_id="call_real", content="done"),
        AssistantChatMessage(content="uncommitted response"),
    ]
    user_index = 1
    end = committed_prefix_end(messages, user_index)
    # user + real tool batch = 4; trailing text assistant not committed.
    assert end == 4, f"Expected end=4, got {end}"
    del messages[end:]
    assert len(messages) == 4
