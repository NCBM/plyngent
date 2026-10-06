"""Thinking-strength resolution across config layers and API surfaces.

``reasoning_effort`` (and the optional Anthropic ``thinking_budget_tokens``) may
be set on ``[agent]``, on a provider, or on a single model. This module merges
the layers — the most specific one wins, per field — and normalizes the result
for the API surface the provider actually talks to, so the agent loop and the
provider bridges receive one value with a single meaning:

* ``""`` — send nothing; the provider applies its own default.
* chat completions / Responses — ``reasoning_effort`` / ``reasoning.effort``.
* Anthropic — ``thinking`` (an effort level has no wire form there).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .routing import model_config_for, resolve_effective_provider

if TYPE_CHECKING:
    from .models import AgentConfig, Provider, ReasoningEffortConfig

# DeepSeek's own ``reasoning_effort`` accepts "max"; OpenAI and Anthropic do not,
# so anywhere else it clamps down to the highest level they do understand.
_DEEPSEEK_CHAT_CONVENTIONS = frozenset({"", "openai"})


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """Merged thinking settings for one provider + model."""

    effort: ReasoningEffortConfig = ""
    # Explicit Anthropic ``thinking`` budget; 0 = derive it from ``effort``.
    thinking_budget_tokens: int = 0


def _clamp_effort(effort: ReasoningEffortConfig, *, preset: str, convention: str) -> ReasoningEffortConfig:
    """Keep ``max`` only on the DeepSeek chat surface that defines it."""
    if effort != "max":
        return effort
    if preset == "deepseek" and convention in _DEEPSEEK_CHAT_CONVENTIONS:
        return effort
    return "xhigh"


def resolve_reasoning(
    agent: AgentConfig | None,
    provider: Provider,
    model: str | None,
) -> ReasoningConfig:
    """Merge ``[agent]`` < provider < model thinking settings for *model*.

    *agent* is ``None`` for callers that only know the provider (config tools,
    tests); the model layer is the provider's entry for *model*, if any. Each
    field falls back on its own, so a model may inherit only the effort (or only
    the budget) from above.
    """
    effective = resolve_effective_provider(provider, model=model)
    model_config = model_config_for(provider, model)

    effort: ReasoningEffortConfig = ""
    for layer in (model_config, provider, agent):
        if layer is not None and layer.reasoning_effort:
            effort = layer.reasoning_effort
            break

    budget = 0
    for layer in (model_config, provider, agent):
        if layer is not None and layer.thinking_budget_tokens > 0:
            budget = layer.thinking_budget_tokens
            break

    return ReasoningConfig(
        effort=_clamp_effort(effort, preset=effective.preset, convention=effective.convention),
        thinking_budget_tokens=budget,
    )


__all__ = ["ReasoningConfig", "resolve_reasoning"]
