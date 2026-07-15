from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.usage import TokenUsage, token_usage_from_api


def test_token_usage_add() -> None:
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    c = a.add(b)
    assert c.prompt_tokens == 13
    assert c.completion_tokens == 7
    assert c.total_tokens == 20
    assert a.prompt_tokens == 10  # immutable-ish via new struct


def test_token_usage_from_api() -> None:
    assert token_usage_from_api(None) is None
    assert token_usage_from_api(UNSET) is None
    assert token_usage_from_api({}) is None
    u = token_usage_from_api({"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
    assert u is not None
    assert u.prompt_tokens == 11
    assert u.completion_tokens == 4
    assert u.total_tokens == 15


def test_token_usage_from_api_infers_total() -> None:
    u = token_usage_from_api({"prompt_tokens": 2, "completion_tokens": 3})
    assert u is not None
    assert u.total_tokens == 5


def test_format_line() -> None:
    line = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3).format_line()
    assert "prompt=1" in line
    assert "completion=2" in line
    assert "total=3" in line
