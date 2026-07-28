from __future__ import annotations

import pytest

from plyngent.config.models import (
    AnthropicProvider,
    DeepseekProvider,
    ModelConfig,
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
    from plyngent.lmproto.openai import OpenAIClient

    client = create_client(provider)
    assert isinstance(client, OpenAIClient)
    assert client.kind == "responses"


def test_openai_provider_tools_passed_to_wrapper() -> None:
    from plyngent.lmproto.openai import OpenAIClient

    provider = OpenAIProvider(
        access_key_or_token="sk-test",
        provider_tools=[{"type": "web_search"}],
    )
    client = create_client(provider)
    assert isinstance(client, OpenAIClient)


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


def test_deepseek_ignores_legacy_convention_extra() -> None:
    provider = DeepseekProvider(
        access_key_or_token="sk-test",
        extras={"convention": "anthropic"},
    )
    client = create_client(provider)
    assert isinstance(client, DeepseekOpenAIClient)


def test_anthropic_not_implemented() -> None:
    from plyngent.lmproto.anthropic import AnthropicClient

    provider = AnthropicProvider(access_key_or_token="sk-test")
    client = create_client(provider)
    assert isinstance(client, AnthropicClient)
    assert client.kind == "messages"


def test_model_level_preset_url_overrides_parent() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://gateway.example/v1",
        models={
            "gpt": ModelConfig(preset="openai", url="https://api.openai.com/v1"),
            "qwen": ModelConfig(),
        },
    )
    openai_client = create_client(provider, model="gpt")
    assert openai_client.kind == "responses"
    assert provider_to_openai_config(provider, model="gpt").base_url == "https://api.openai.com/v1"

    compat_client = create_client(provider, model="qwen")
    assert isinstance(compat_client, OpenAICompatibleClient)
    assert provider_to_openai_config(provider, model="qwen").base_url == "https://gateway.example/v1"


def test_model_level_anthropic_preset() -> None:
    from plyngent.lmproto.anthropic import AnthropicClient

    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://gateway.example/v1",
        models={"claude": ModelConfig(preset="anthropic", url="https://api.anthropic.com/v1")},
    )
    client = create_client(provider, model="claude")
    assert isinstance(client, AnthropicClient)
    assert client.kind == "messages"
