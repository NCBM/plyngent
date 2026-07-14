from __future__ import annotations

from plyngent.agent.budget import estimate_message_chars, truncate_tool_result
from plyngent.lmproto.openai_compatible.model import ToolChatMessage, UserChatMessage


def test_truncate_tool_result_short() -> None:
    assert truncate_tool_result("hello", 100) == "hello"


def test_truncate_tool_result_long() -> None:
    text = "a" * 50
    out = truncate_tool_result(text, 20)
    assert out.startswith("a" * 20)
    assert "truncated" in out
    assert "30" in out


def test_estimate_message_chars() -> None:
    assert estimate_message_chars(UserChatMessage(content="hello")) == 5  # noqa: PLR2004
    tool = ToolChatMessage(content="abc", tool_call_id="id1")
    assert estimate_message_chars(tool) == 6  # noqa: PLR2004
