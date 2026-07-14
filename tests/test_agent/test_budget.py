from __future__ import annotations

from plyngent.agent.budget import truncate_tool_result


def test_truncate_tool_result_short() -> None:
    assert truncate_tool_result("hello", 100) == "hello"


def test_truncate_tool_result_long() -> None:
    text = "a" * 50
    out = truncate_tool_result(text, 20)
    assert out.startswith("a" * 20)
    assert "truncated" in out
    assert "30" in out
