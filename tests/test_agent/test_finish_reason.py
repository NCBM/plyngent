"""Tests for the canonical finish_reason vocabulary shared by provider bridges."""

from __future__ import annotations

import pytest

from plyngent.agent.finish_reason import chat_finish_reason


@pytest.mark.parametrize(
    ("raw", "has_tool_calls", "expected"),
    [
        (None, False, "stop"),
        (None, True, "tool_calls"),
        ("", False, "stop"),
        ("end_turn", False, "stop"),
        ("stop_sequence", False, "stop"),
        ("completed", False, "stop"),
        ("stop", False, "stop"),
        ("max_tokens", False, "length"),
        ("max_output_tokens", False, "length"),
        ("length", False, "length"),
        ("content_filter", False, "content_filter"),
        ("refusal", False, "content_filter"),
        ("tool_use", False, "tool_calls"),
        ("tool_calls", False, "tool_calls"),
        ("Tool_Use", False, "tool_calls"),
        ("MAX_TOKENS", False, "length"),
    ],
)
def test_chat_finish_reason(raw: str | None, has_tool_calls: bool, expected: str) -> None:
    assert chat_finish_reason(raw, has_tool_calls=has_tool_calls) == expected
