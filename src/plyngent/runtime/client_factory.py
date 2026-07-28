from __future__ import annotations

import math
from typing import TYPE_CHECKING

from plyngent.config.routing import EffectiveProvider, resolve_effective_provider
from plyngent.lmproto.anthropic import AnthropicClient
from plyngent.lmproto.anthropic.config import AnthropicConfig as AnthropicConfigCls
from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai import OpenAIClient
from plyngent.lmproto.openai_compatible import OpenAICompatibleClient, OpenAIConfig
from plyngent.lmproto.openai_compatible.config import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_READ_TIMEOUT,
    HttpTimeout,
)

if TYPE_CHECKING:
    from plyngent.config.models import HttpTimeoutConfig, Provider

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

type ProtocolClient = OpenAIClient | OpenAICompatibleClient | DeepseekOpenAIClient | AnthropicClient
# Backward-compatible name used by older imports/tests.
type OpenAICompatibleClientUnion = ProtocolClient


class ProviderNotSupportedError(NotImplementedError):
    """Raised when a provider preset cannot be turned into a runtime client."""


class InvalidHttpTimeoutError(ValueError):
    """Raised when a provider ``timeout`` value is not usable for HTTP clients."""


def normalize_http_timeout(timeout: float | HttpTimeoutConfig | None) -> HttpTimeout:
    """Normalize TOML/provider timeout into a niquests session timeout.

    * ``None`` → product defaults ``(connect=10, read=600)``
    * ``float`` / ``int`` → single timeout (niquests applies to the request)
    * :class:`HttpTimeoutConfig` → ``(connect, read)`` with defaults for omitted fields

    All finite values must be ``> 0``.
    """
    if timeout is None:
        return (DEFAULT_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT)
    if isinstance(timeout, bool):
        # ``bool`` is an ``int`` subclass; reject before the numeric branch.
        msg = "timeout must be a positive number or { connect, read }, not a boolean"
        raise InvalidHttpTimeoutError(msg)
    if isinstance(timeout, int | float):
        value = float(timeout)
        if not math.isfinite(value) or value <= 0:
            msg = f"timeout must be a finite number > 0, got {timeout!r}"
            raise InvalidHttpTimeoutError(msg)
        return value

    connect = DEFAULT_HTTP_CONNECT_TIMEOUT if timeout.connect is None else float(timeout.connect)
    read = DEFAULT_HTTP_READ_TIMEOUT if timeout.read is None else float(timeout.read)
    if not math.isfinite(connect) or connect <= 0:
        msg = f"timeout.connect must be a finite number > 0, got {timeout.connect!r}"
        raise InvalidHttpTimeoutError(msg)
    if not math.isfinite(read) or read <= 0:
        msg = f"timeout.read must be a finite number > 0, got {timeout.read!r}"
        raise InvalidHttpTimeoutError(msg)
    return (connect, read)


def provider_to_openai_config(
    provider: Provider | EffectiveProvider,
    *,
    model: str | None = None,
) -> OpenAIConfig:
    """Map provider/effective config to the shared OpenAI-compatible HTTP config."""
    if isinstance(provider, EffectiveProvider):
        effective = provider
    else:
        effective = resolve_effective_provider(provider, model=model)
    if not effective.url and effective.preset in {"openai-compatible", "deepseek"}:
        msg = f"{effective.preset} provider requires a non-empty url"
        raise ProviderNotSupportedError(msg)
    try:
        http_timeout = normalize_http_timeout(effective.timeout)
    except InvalidHttpTimeoutError as exc:
        raise ProviderNotSupportedError(str(exc)) from exc
    return OpenAIConfig(
        access_key_or_token=effective.access_key_or_token,
        base_url=effective.url,
        timeout=http_timeout,
    )


def create_client(provider: Provider, *, model: str | None = None) -> ProtocolClient:
    """Build a protocol client for *provider*, applying model-level routing.

    ``preset`` always decides API conventions: ``openai`` → /responses,
    ``openai-compatible`` → /chat/completions, ``anthropic`` → /messages.
    """
    effective = resolve_effective_provider(provider, model=model)
    if effective.preset == "openai":
        return OpenAIClient(provider_to_openai_config(effective))
    if effective.preset == "openai-compatible":
        return OpenAICompatibleClient(provider_to_openai_config(effective))
    if effective.preset == "deepseek":
        return DeepseekOpenAIClient(provider_to_openai_config(effective))
    if effective.preset == "anthropic":
        return AnthropicClient(
            AnthropicConfigCls(
                api_key=effective.access_key_or_token,
                base_url=effective.url or DEFAULT_ANTHROPIC_BASE_URL,
            )
        )
    msg = f"provider preset {effective.preset!r} is not supported"
    raise ProviderNotSupportedError(msg)
