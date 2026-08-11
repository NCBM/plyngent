from __future__ import annotations

from plyngent.agent.budget import estimate_message_chars, truncate_tool_result
from plyngent.lmproto.openai_compatible.model import ToolChatMessage, UserChatMessage


def test_truncate_tool_result_short() -> None:
    assert truncate_tool_result("hello", 100) == "hello"


def test_truncate_tool_result_long() -> None:
    text = "a" * 50
    out = truncate_tool_result(text, 20)
    assert "Truncated" in out
    assert "20" in out
    assert "30" in out
    # Generic tool results now embed a memory token too (resumable via get_truncated).
    assert "truncate_token=" in out
    assert out.startswith(("a" * 10, "a" * 20))  # marker is budgeted


def test_truncate_tool_result_fits_no_marker() -> None:
    assert truncate_tool_result("tiny", 100) == "tiny"
    assert "Truncated" not in truncate_tool_result("tiny", 100)


def test_estimate_message_chars() -> None:
    assert estimate_message_chars(UserChatMessage(content="hello")) == 5
    tool = ToolChatMessage(content="abc", tool_call_id="id1")
    assert estimate_message_chars(tool) == 6
