"""Tests for synthetic_tool todo nag behaviour on retry.

Tests the core rollback and _synthetic_todo_pair_after logic.
"""

from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.chat import (
    _synthetic_todo_pair_after,
    committed_prefix_end,
)
from plyngent.agent.todo_nag import (
    _append_synthetic_todo_list,
    is_synthetic_todo_nag_call_id,
    refresh_synthetic_todo_nags,
)
from plyngent.agent.todo_stack import TodoStack
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)


def test_synthetic_pair_after_find_at_index() -> None:
    """_synthetic_todo_pair_after correctly identifies a pair at index+1/+2."""
    stack = TodoStack()
    _ = stack.push("do this")

    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    _append_synthetic_todo_list(messages, stack.render())

    assert _synthetic_todo_pair_after(messages, 1), "Should find synthetic pair at user_index=1"

    # When user is at different index, same pair should NOT be found
    assert not _synthetic_todo_pair_after(messages, 0), "Should NOT find pair at user_index=0 (pair is at 2/3)"


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
    assert _synthetic_todo_pair_after(messages, user_index)


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

    # Pair should survive
    assert _synthetic_todo_pair_after(messages, user_index)


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

    assert _synthetic_todo_pair_after(messages, user_index)

    # Simulate second failure: roll back again
    end = committed_prefix_end(messages, user_index)
    del messages[end:]

    assert _synthetic_todo_pair_after(messages, user_index)

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


def test_retry_skips_injection_when_pair_present() -> None:
    """Simulating retry logic: should skip injection when pair exists."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    _append_synthetic_todo_list(messages, "task")

    stack = TodoStack()
    _ = stack.push("task")

    user_index = 1

    # Refresh synthetic nags (as _run_from_user_message does)
    _ = refresh_synthetic_todo_nags(messages, stack)

    should_inject = not _synthetic_todo_pair_after(messages, user_index)

    assert not should_inject, "Should NOT inject on retry when pair exists"

    # For developer and user strategies, the outer nag check passes
    # (injects) regardless of pair existence — the pair_exists branch is
    # only exercised by synthetic_tool.
    assert _synthetic_todo_pair_after(messages, user_index), "pair should still exist"


def test_synthetic_pair_after_empty_list() -> None:
    """_synthetic_todo_pair_after returns False for short lists."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
    ]
    assert not _synthetic_todo_pair_after(messages, 1)
    assert not _synthetic_todo_pair_after(messages, 0)

    # Edge: user at end with no room for pair
    assert not _synthetic_todo_pair_after(messages, 0)


def test_synthetic_pair_after_non_matching() -> None:
    """_synthetic_todo_pair_after returns False for non-synthetic content."""
    messages: list[AnyChatMessage] = [
        SystemChatMessage(content="sys"),
        UserChatMessage(content="hi"),
        AssistantChatMessage(content="normal response"),
    ]
    # messages[2] is AssistantChatMessage but has no tool_calls
    assert not _synthetic_todo_pair_after(messages, 1)

    # Real tool call instead of synthetic
    messages.append(
        AssistantChatMessage(
            content=UNSET,
            tool_calls=[
                AssistantFunctionToolCall(
                    id="call_real",
                    function=AssistantFunctionTool(name="tool", arguments="{}"),
                )
            ],
        )
    )
    messages.append(ToolChatMessage(tool_call_id="call_real", content="done"))
    assert not _synthetic_todo_pair_after(messages, 1), "Real tool calls should not match"
