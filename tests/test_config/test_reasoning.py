"""Tests for thinking-strength resolution across config layers and surfaces."""

from __future__ import annotations

from plyngent.config.models import (
    AgentConfig,
    AnthropicProvider,
    DeepseekProvider,
    ModelConfig,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from plyngent.config.reasoning import resolve_reasoning


def _compat() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(access_key_or_token="sk-test", url="https://example.com/v1")


def test_agent_layer_only() -> None:
    reasoning = resolve_reasoning(
        AgentConfig(reasoning_effort="high", thinking_budget_tokens=9000),
        _compat(),
        "gpt-4o-mini",
    )
    assert reasoning.effort == "high"
    assert reasoning.thinking_budget_tokens == 9000


def test_provider_overrides_agent_and_model_overrides_provider() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        reasoning_effort="low",
        models={"gpt-4o-mini": ModelConfig(reasoning_effort="xhigh")},
    )
    agent = AgentConfig(reasoning_effort="minimal")
    assert resolve_reasoning(agent, provider, "gpt-4o-mini").effort == "xhigh"
    assert resolve_reasoning(agent, provider, "other-model").effort == "low"
    assert resolve_reasoning(agent, provider, None).effort == "low"


def test_fields_fall_back_independently() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        reasoning_effort="high",
        thinking_budget_tokens=20000,
        models={"gpt-4o-mini": ModelConfig(thinking_budget_tokens=4000)},
    )
    reasoning = resolve_reasoning(AgentConfig(reasoning_effort="low"), provider, "gpt-4o-mini")
    # Model sets the budget only → the effort still comes from the provider.
    assert reasoning.effort == "high"
    assert reasoning.thinking_budget_tokens == 4000


def test_agent_layer_optional() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        reasoning_effort="medium",
    )
    assert resolve_reasoning(None, provider, "gpt-4o-mini").effort == "medium"


def test_unset_layers_stay_unspecified() -> None:
    reasoning = resolve_reasoning(None, _compat(), "gpt-4o-mini")
    assert reasoning.effort == ""
    assert reasoning.thinking_budget_tokens == 0


def test_non_positive_budget_is_ignored() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        thinking_budget_tokens=0,
    )
    resolved = resolve_reasoning(AgentConfig(thinking_budget_tokens=-5), provider, "gpt-4o-mini")
    assert resolved.thinking_budget_tokens == 0


def test_max_survives_only_on_deepseek_chat() -> None:
    chat = DeepseekProvider(access_key_or_token="sk-test", reasoning_effort="max")
    assert resolve_reasoning(None, chat, "deepseek-v4-pro").effort == "max"

    responses = DeepseekProvider(access_key_or_token="sk-test", convention="responses", reasoning_effort="max")
    assert resolve_reasoning(None, responses, "deepseek-v4-pro").effort == "xhigh"

    anthropic = AnthropicProvider(access_key_or_token="sk-test", reasoning_effort="max")
    assert resolve_reasoning(None, anthropic, "claude-test").effort == "xhigh"

    openai = OpenAIProvider(access_key_or_token="sk-test", reasoning_effort="max")
    assert resolve_reasoning(None, openai, "gpt-5.4").effort == "xhigh"


def test_effort_levels_pass_through_unchanged() -> None:
    provider = _compat()
    for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
        resolved = resolve_reasoning(AgentConfig(reasoning_effort=effort), provider, "gpt-4o-mini")
        assert resolved.effort == effort
