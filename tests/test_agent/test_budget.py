from __future__ import annotations

from plyngent.agent.budget import (
    cap_hard_tool_text,
    cap_tool_message,
    estimate_message_chars,
    truncate_for_request,
)
from plyngent.lmproto.openai_compatible.model import ToolChatMessage, UserChatMessage


def test_cap_hard_tool_text_short_untouched() -> None:
    assert cap_hard_tool_text("hello", 100) == "hello"


def test_cap_hard_tool_text_cuts_with_ceiling_marker() -> None:
    out = cap_hard_tool_text("a" * 50, 20)
    assert out.startswith("a" * 20)
    assert "[Stored at the 20-char ceiling: 30 chars beyond it]" in out


def test_cap_tool_message_names_the_document() -> None:
    message = ToolChatMessage(content="a" * 50, tool_call_id="call-7")
    capped = cap_tool_message(message, 20)
    assert capped is not message
    assert capped.tool_call_id == "call-7"
    assert "document_id=call-7" in capped.content
    assert capped.content.startswith("a")
    # The stored message keeps its body whole.
    assert message.content == "a" * 50


def test_cap_tool_message_short_untouched() -> None:
    message = ToolChatMessage(content="tiny", tool_call_id="call-1")
    assert cap_tool_message(message, 100) is message


def test_truncate_for_request_caps_only_tool_messages() -> None:
    user = UserChatMessage(content="u" * 1000)
    tool = ToolChatMessage(content="t" * 1000, tool_call_id="call-1")
    out = truncate_for_request([user, tool], max_tool_result_chars=200)
    assert out[0] is user
    capped = out[1]
    assert isinstance(capped, ToolChatMessage)
    assert "document_id=call-1" in capped.content
    assert len(capped.content) <= 200


def test_estimate_message_chars() -> None:
    assert estimate_message_chars(UserChatMessage(content="hello")) == 5
    tool = ToolChatMessage(content="abc", tool_call_id="id1")
    assert estimate_message_chars(tool) == 6
