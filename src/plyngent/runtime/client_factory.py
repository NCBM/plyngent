from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config.models import (
    DeepseekProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
    Provider,
)
from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai import OpenAIClient
from plyngent.lmproto.openai_compatible import OpenAICompatibleClient, OpenAIConfig

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.agent.responses_client import ResponsesChatClient

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

type ProtocolClient = (
    OpenAIClient | OpenAICompatibleClient | DeepseekOpenAIClient | ResponsesChatClient
)
# Backward-compatible name used by older imports/tests.
type OpenAICompatibleClientUnion = ProtocolClient


class ProviderNotSupportedError(NotImplementedError):
    """Raised when a provider preset cannot be turned into a runtime client."""


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
    return OpenAIConfig(
        access_key_or_token=provider.access_key_or_token,
        base_url=base_url,
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

        return wrap_openai_for_agent(OpenAIClient(provider_to_openai_config(provider)))
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
