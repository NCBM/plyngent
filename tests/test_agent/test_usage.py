from __future__ import annotations

from msgspec import UNSET

from plyngent.agent.usage import (
    TokenUsage,
    chars_to_tokens,
    estimate_token_usage,
    resolve_round_usage,
    token_usage_from_api,
)
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    UserChatMessage,
)


def test_token_usage_add() -> None:
    a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, source="api")
    b = TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5, source="api")
    c = a.add(b)
    assert c.prompt_tokens == 13
    assert c.completion_tokens == 7
    assert c.total_tokens == 20
    assert c.source == "api"
    assert a.prompt_tokens == 10


def test_token_usage_add_mixed_source() -> None:
    a = TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1, source="api")
    b = TokenUsage(prompt_tokens=2, completion_tokens=0, total_tokens=2, source="estimate")
    assert a.add(b).source == "mixed"


def test_token_usage_from_api() -> None:
    assert token_usage_from_api(None) is None
    assert token_usage_from_api(UNSET) is None
    assert token_usage_from_api({}) is None
    u = token_usage_from_api({"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15})
    assert u is not None
    assert u.prompt_tokens == 11
    assert u.completion_tokens == 4
    assert u.total_tokens == 15
    assert u.source == "api"


def test_token_usage_from_api_infers_total() -> None:
    u = token_usage_from_api({"prompt_tokens": 2, "completion_tokens": 3})
    assert u is not None
    assert u.total_tokens == 5


def test_token_usage_from_api_responses_fields() -> None:
    u = token_usage_from_api({"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})
    assert u is not None
    assert u.prompt_tokens == 7
    assert u.completion_tokens == 2
    assert u.total_tokens == 9


def test_chars_to_tokens() -> None:
    assert chars_to_tokens(0) == 0
    assert chars_to_tokens(1) == 1
    assert chars_to_tokens(4) == 1
    assert chars_to_tokens(5) == 2
    assert chars_to_tokens(8) == 2
    assert chars_to_tokens(9) == 3


def test_estimate_token_usage() -> None:
    # 8 chars prompt → 2 tokens at 4 cpt; 4 chars completion → 1 token
    usage = estimate_token_usage(
        [UserChatMessage(content="12345678")],
        AssistantChatMessage(content="abcd"),
    )
    assert usage.source == "estimate"
    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 1
    assert usage.total_tokens == 3


def test_resolve_round_usage_prefers_api() -> None:
    usage = resolve_round_usage(
        {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        [UserChatMessage(content="x")],
        AssistantChatMessage(content="y"),
    )
    assert usage.source == "api"
    assert usage.prompt_tokens == 100


def test_resolve_round_usage_falls_back_to_estimate() -> None:
    usage = resolve_round_usage(
        None,
        [UserChatMessage(content="12345678")],
        AssistantChatMessage(content="abcd"),
    )
    assert usage.source == "estimate"
    assert usage.total_tokens == 3


def test_format_line_marks_estimate() -> None:
    line = TokenUsage(prompt_tokens=1, completion_tokens=2, total_tokens=3, source="estimate").format_line()
    assert "prompt=1" in line
    assert "(est)" in line
    mixed = TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1, source="mixed").format_line()
    assert "(api+est)" in mixed
    billed = TokenUsage(prompt_tokens=1, completion_tokens=0, total_tokens=1).format_line(billed=True)
    assert billed.startswith("billed tokens")
