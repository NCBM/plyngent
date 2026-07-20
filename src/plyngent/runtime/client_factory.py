from __future__ import annotations

import math
from typing import TYPE_CHECKING

from plyngent.config.models import (
    DeepseekProvider,
    HttpTimeoutConfig,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
)
from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai import OpenAIClient
from plyngent.lmproto.openai_compatible import OpenAICompatibleClient, OpenAIConfig
from plyngent.lmproto.openai_compatible.config import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_READ_TIMEOUT,
    HttpTimeout,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.agent.responses_client import ResponsesChatClient

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

type ProtocolClient = OpenAIClient | OpenAICompatibleClient | DeepseekOpenAIClient | ResponsesChatClient
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

    # Remaining union member: HttpTimeoutConfig
    connect = DEFAULT_HTTP_CONNECT_TIMEOUT if timeout.connect is None else float(timeout.connect)
    read = DEFAULT_HTTP_READ_TIMEOUT if timeout.read is None else float(timeout.read)
    if not math.isfinite(connect) or connect <= 0:
        msg = f"timeout.connect must be a finite number > 0, got {timeout.connect!r}"
        raise InvalidHttpTimeoutError(msg)
    if not math.isfinite(read) or read <= 0:
        msg = f"timeout.read must be a finite number > 0, got {timeout.read!r}"
        raise InvalidHttpTimeoutError(msg)
    return (connect, read)


def provider_to_openai_config(provider: OpenAIProvider | OpenAICompatibleProvider | DeepseekProvider) -> OpenAIConfig:
    """Map a provider config entry to :class:`OpenAIConfig`."""
    if isinstance(provider, OpenAIProvider):
        base_url = provider.url or DEFAULT_OPENAI_BASE_URL
    elif isinstance(provider, DeepseekProvider):
        base_url = provider.url or DEFAULT_DEEPSEEK_BASE_URL
    else:
        base_url = provider.url
        if not base_url:
            msg = "openai-compatible provider requires a non-empty url"
            raise ProviderNotSupportedError(msg)
    try:
        http_timeout = normalize_http_timeout(provider.timeout)
    except InvalidHttpTimeoutError as exc:
        raise ProviderNotSupportedError(str(exc)) from exc
    return OpenAIConfig(
        access_key_or_token=provider.access_key_or_token,
        base_url=base_url,
        timeout=http_timeout,
    )


def _deepseek_convention(extras: Mapping[str, str]) -> str:
    return extras.get("convention", "openai").lower()


def create_client(provider: Provider) -> ProtocolClient:
    """Build a protocol client for the given provider config entry.

    OpenAI platform providers are wrapped so the agent uses the Responses API
    while still exposing a chat-completions-shaped interface.

    Raises:
        ProviderNotSupportedError: When the provider preset (or DeepSeek convention)
            has no implemented client yet.
    """
    if isinstance(provider, OpenAIProvider):
        from plyngent.agent.responses_client import wrap_openai_for_agent

        return wrap_openai_for_agent(
            OpenAIClient(provider_to_openai_config(provider)),
            provider_tools=provider.provider_tools or None,
        )
    if isinstance(provider, OpenAICompatibleProvider):
        return OpenAICompatibleClient(provider_to_openai_config(provider))
    if isinstance(provider, DeepseekProvider):
        convention = _deepseek_convention(provider.extras)
        if convention in {"openai", "openai_compat", "openai-compatible"}:
            return DeepseekOpenAIClient(provider_to_openai_config(provider))
        if convention == "anthropic":
            msg = "deepseek anthropic convention is not implemented"
            raise ProviderNotSupportedError(msg)
        msg = f"unknown deepseek convention {convention!r}"
        raise ProviderNotSupportedError(msg)
    # Remaining Provider variant: AnthropicProvider
    msg = "anthropic provider client is not implemented"
    raise ProviderNotSupportedError(msg)
