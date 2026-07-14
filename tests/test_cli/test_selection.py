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


def test_select_model_prompt_when_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAIProvider(access_key_or_token="sk")

    def _prompt(*_args: object, **_kwargs: object) -> str:
        return "gpt-test"

    monkeypatch.setattr("click.prompt", _prompt)
    assert select_model(provider) == "gpt-test"
