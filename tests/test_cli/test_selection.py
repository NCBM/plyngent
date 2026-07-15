from __future__ import annotations

import pytest

from plyngent.cli.selection import select_model, select_provider
from plyngent.config.models import ModelConfig, OpenAICompatibleProvider, OpenAIProvider


def test_select_provider_preferred() -> None:
    providers = {
        "a": OpenAIProvider(access_key_or_token="sk"),
        "b": OpenAICompatibleProvider(access_key_or_token="sk", url="https://x/v1"),
    }
    name, provider = select_provider(providers, preferred="b")
    assert name == "b"
    assert isinstance(provider, OpenAICompatibleProvider)


def test_select_provider_single_auto() -> None:
    providers = {"only": OpenAIProvider(access_key_or_token="sk")}
    name, _ = select_provider(providers)
    assert name == "only"


def test_select_provider_unknown() -> None:
    providers = {"a": OpenAIProvider(access_key_or_token="sk")}
    with pytest.raises(Exception, match="unknown provider"):
        _ = select_provider(providers, preferred="nope")


def test_select_model_from_list() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"m1": ModelConfig()},
    )
    assert select_model(provider) == "m1"
    assert select_model(provider, preferred="m1") == "m1"


def test_select_model_prompt_when_empty() -> None:
    from plyngent.prompting import temporary_backend
    from tests.test_prompting import ScriptedBackend

    provider = OpenAIProvider(access_key_or_token="sk")
    with temporary_backend(ScriptedBackend(["gpt-test"])):
        assert select_model(provider) == "gpt-test"


def test_select_provider_interactive_choose() -> None:
    from plyngent.prompting import temporary_backend
    from tests.test_prompting import ScriptedBackend

    providers = {
        "a": OpenAIProvider(access_key_or_token="sk", models={"m": ModelConfig()}),
        "b": OpenAICompatibleProvider(
            access_key_or_token="sk",
            url="https://x/v1",
            models={"m": ModelConfig()},
        ),
    }
    with temporary_backend(ScriptedBackend(["2"])):
        name, _ = select_provider(providers)
    assert name == "b"


def test_select_model_when_preferred_missing_raises() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"m1": ModelConfig()},
    )
    with pytest.raises(Exception, match="unknown model"):
        _ = select_model(provider, preferred="nope")
