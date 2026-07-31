from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from .models import (
    AnthropicProvider,
    DeepseekProvider,
    HttpTimeoutConfig,
    ModelConfig,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


type ProviderPreset = Literal["openai", "openai-compatible", "anthropic", "deepseek"]

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"


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
    # DeepSeek API surface (see DeepSeekConvention): "openai" / "anthropic" /
    # "responses" for deepseek presets; "" (unused) for other presets.
    convention: str = ""


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
        return DEFAULT_DEEPSEEK_BASE_URL
    # Generic OpenAI-compatible providers usually require a host-specific URL.
    return ""


def _default_url_for_deepseek_convention(convention: str) -> str:
    """Default DeepSeek base URL for an API convention.

    The Anthropic-convention endpoint lives under ``/anthropic``; chat and
    Responses share the plain ``https://api.deepseek.com`` base.
    """
    if convention == "anthropic":
        return DEFAULT_DEEPSEEK_ANTHROPIC_BASE_URL
    return DEFAULT_DEEPSEEK_BASE_URL


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
    Exception: a ``deepseek`` preset may pick the Responses or Anthropic
    surface via the ``convention`` field (``"responses"`` / ``"anthropic"``);
    model-level ``convention`` wins over the provider-level one.
    """
    parent_preset = provider_preset(provider)
    model_cfg = model_config_for(provider, model)
    if model_cfg is not None and model_cfg.preset.strip():
        preset = validate_model_preset(model_cfg.preset.strip())
    else:
        preset = parent_preset

    # DeepSeek API surface: model override > provider convention > chat default.
    convention = ""
    if preset == "deepseek" and isinstance(provider, DeepseekProvider):
        convention = provider.convention or "openai"
        if model_cfg is not None and model_cfg.convention:
            convention = model_cfg.convention

    url = model_cfg.url.strip() if model_cfg is not None and model_cfg.url.strip() else provider.url.strip()
    if not url:
        if preset == "deepseek":
            url = _default_url_for_deepseek_convention(convention)
        else:
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
        convention=convention,
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
        "convention": effective.convention,
    }
