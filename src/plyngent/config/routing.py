from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from .models import (
    AnthropicProvider,
    HttpTimeoutConfig,
    ModelConfig,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


type ProviderPreset = Literal["openai", "openai-compatible", "anthropic", "deepseek"]


@dataclass(frozen=True, slots=True)
class EffectiveProvider:
    """Provider routing after applying model-level ``preset`` / ``url`` overrides."""

    preset: ProviderPreset
    access_key_or_token: str
    url: str
    model: str | None
    timeout: float | HttpTimeoutConfig | None
    provider_tools: list[dict[str, object]]
    source: Provider


def provider_preset(provider: Provider) -> ProviderPreset:
    """Return the tagged-union preset for *provider*."""
    if isinstance(provider, OpenAIProvider):
        return "openai"
    if isinstance(provider, OpenAICompatibleProvider):
        return "openai-compatible"
    if isinstance(provider, AnthropicProvider):
        return "anthropic"
    return "deepseek"


def default_url_for_preset(preset: ProviderPreset) -> str:
    """Default base URL; includes the API version path when applicable."""
    if preset == "openai":
        return "https://api.openai.com/v1"
    if preset == "anthropic":
        return "https://api.anthropic.com/v1"
    if preset == "deepseek":
        return "https://api.deepseek.com"
    # Generic OpenAI-compatible providers usually require a host-specific URL.
    return ""


def validate_model_preset(preset: str) -> ProviderPreset:
    if preset in {"openai", "openai-compatible", "anthropic", "deepseek"}:
        return cast("ProviderPreset", preset)
    msg = f"unsupported model preset {preset!r}"
    raise ValueError(msg)


def model_config_for(provider: Provider, model: str | None) -> ModelConfig | None:
    if not model:
        return None
    return provider.models.get(model)


def resolve_effective_provider(provider: Provider, *, model: str | None = None) -> EffectiveProvider:
    """Apply model-level preset/url overrides to *provider*.

    ``preset`` always decides API conventions: ``openai`` means Responses,
    ``openai-compatible`` means Chat Completions, ``anthropic`` means Messages.
    """
    parent_preset = provider_preset(provider)
    model_cfg = model_config_for(provider, model)
    if model_cfg is not None and model_cfg.preset.strip():
        preset = validate_model_preset(model_cfg.preset.strip())
    else:
        preset = parent_preset
    url = model_cfg.url.strip() if model_cfg is not None and model_cfg.url.strip() else provider.url.strip()
    if not url:
        url = default_url_for_preset(preset)

    provider_tools: list[dict[str, object]] = []
    if preset == "openai" and isinstance(provider, OpenAIProvider) and parent_preset == "openai":
        provider_tools = [dict(item) for item in provider.provider_tools]

    return EffectiveProvider(
        preset=preset,
        access_key_or_token=provider.access_key_or_token,
        url=url,
        model=model,
        timeout=provider.timeout,
        provider_tools=provider_tools,
        source=provider,
    )


def effective_provider_to_config_dict(effective: EffectiveProvider) -> Mapping[str, object]:
    """Debug/status-friendly dict for an effective provider."""
    return {
        "preset": effective.preset,
        "access_key_or_token": effective.access_key_or_token,
        "url": effective.url,
        "model": effective.model or "",
        "timeout": effective.timeout,
        "provider_tools": effective.provider_tools,
    }
