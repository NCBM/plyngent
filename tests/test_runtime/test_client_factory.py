from __future__ import annotations

import pytest

from plyngent.config.models import (
    AnthropicProvider,
    DeepseekProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from plyngent.lmproto.deepseek import DeepseekOpenAIClient
from plyngent.lmproto.openai import OpenAIClient
from plyngent.lmproto.openai_compatible import OpenAICompatibleClient
from plyngent.runtime import ProviderNotSupportedError, create_client, provider_to_openai_config


def test_openai_provider_defaults_base_url() -> None:
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = provider_to_openai_config(provider)
    assert config.access_key_or_token == "sk-test"
    assert config.base_url == "https://api.openai.com/v1"
    from plyngent.agent.responses_client import ResponsesChatClient

    client = create_client(provider)
    assert isinstance(client, ResponsesChatClient)
    assert hasattr(client, "chat_completions")
    assert client._provider_tools == []


def test_openai_provider_tools_passed_to_wrapper() -> None:
    from plyngent.agent.responses_client import ResponsesChatClient

    provider = OpenAIProvider(
        access_key_or_token="sk-test",
        provider_tools=[{"type": "web_search"}],
    )
    client = create_client(provider)
    assert isinstance(client, ResponsesChatClient)
    assert client._provider_tools == [{"type": "web_search"}]


def test_openai_compatible_requires_url() -> None:
    provider = OpenAICompatibleProvider(access_key_or_token="sk-test")
    with pytest.raises(ProviderNotSupportedError, match="url"):
        _ = create_client(provider)


def test_openai_compatible_client() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
    )
    client = create_client(provider)
    assert isinstance(client, OpenAICompatibleClient)
    assert not isinstance(client, OpenAIClient)
    assert provider_to_openai_config(provider).base_url == "https://example.com/v1"


def test_deepseek_openai_convention() -> None:
    provider = DeepseekProvider(access_key_or_token="sk-test")
    client = create_client(provider)
    assert isinstance(client, DeepseekOpenAIClient)
    assert provider_to_openai_config(provider).base_url == "https://api.deepseek.com"
    assert "deepseek-v4-flash" in provider.models
    assert "deepseek-v4-pro" in provider.models


def test_deepseek_anthropic_convention_not_implemented() -> None:
    provider = DeepseekProvider(
        access_key_or_token="sk-test",
        extras={"convention": "anthropic"},
    )
    with pytest.raises(ProviderNotSupportedError, match="anthropic"):
        _ = create_client(provider)


def test_anthropic_not_implemented() -> None:
    provider = AnthropicProvider(access_key_or_token="sk-test")
    with pytest.raises(ProviderNotSupportedError, match="anthropic"):
        _ = create_client(provider)
