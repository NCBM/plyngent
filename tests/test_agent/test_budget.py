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


def test_cap_hard_tool_text_cuts_with_the_notice_up_front() -> None:
    """The notice leads: the wire cap sends the head, so a trailing one is lost."""
    out = cap_hard_tool_text("a" * 50, 20)
    head, _, body = out.partition("\n")
    assert "20-char ceiling" in head
    assert "30 chars beyond it were dropped and cannot be read back" in head
    assert body == "a" * 20


def test_cap_tool_message_names_the_document() -> None:
    message = ToolChatMessage(content="a" * 50, tool_call_id="call-7")
    capped = cap_tool_message(message, 20)
    assert capped is not message
    assert capped.tool_call_id == "call-7"
    assert "document_id=call-7" in capped.content
    assert capped.content.startswith("a")
    assert "storage ceiling" not in capped.content
    # The stored message keeps its body whole.
    assert message.content == "a" * 50


def test_cap_tool_message_flags_a_body_the_ceiling_cut() -> None:
    """A cut body is not the whole output, so the marker must not imply it is."""
    stored = cap_hard_tool_text("a" * 500, 100)
    capped = cap_tool_message(ToolChatMessage(content=stored, tool_call_id="call-9"), 60)
    assert "document_id=call-9" in capped.content
    assert "cut at the storage ceiling" in capped.content


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
