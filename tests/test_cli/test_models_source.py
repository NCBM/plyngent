from __future__ import annotations

import pytest

from plyngent.cli.models_source import (
    client_supports_models,
    config_model_ids,
    fetch_remote_model_ids,
    merge_model_choices,
    model_choices_for_provider,
    needs_remote_models_for_selection,
)
from plyngent.config.models import ModelConfig, OpenAICompatibleProvider


def test_merge_model_choices_remote_first() -> None:
    # remote-first: remote sorted, then config-only
    assert merge_model_choices(["b", "a"], ["a", "c"]) == ["a", "c", "b"]
    assert merge_model_choices(["a"], None) == ["a"]
    assert merge_model_choices([], ["z"]) == ["z"]
    assert merge_model_choices(["cfg"], ["remote", "cfg"], prefer="remote") == ["cfg", "remote"]
    assert merge_model_choices(["b", "a"], ["a", "c"], prefer="union") == ["a", "b", "c"]
    assert merge_model_choices(["b", "a"], ["a", "c"], prefer="config") == ["a", "b", "c"]


def test_model_choices_for_provider() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"cfg": ModelConfig()},
    )
    assert config_model_ids(provider) == ["cfg"]
    assert model_choices_for_provider(provider, remote_ids=["remote", "cfg"]) == ["cfg", "remote"]
    assert model_choices_for_provider(provider, remote_ids=["remote"]) == ["remote", "cfg"]


def test_client_supports_models() -> None:
    class Ok:
        async def models(self) -> list[str]:
            return ["m"]

    class No:
        pass

    assert client_supports_models(Ok())
    assert not client_supports_models(No())


@pytest.mark.asyncio
async def test_fetch_remote_model_ids() -> None:
    class Ok:
        async def models(self) -> list[str]:
            return ["z", "a"]

    assert await fetch_remote_model_ids(Ok()) == ["z", "a"]

    with pytest.raises(TypeError, match="does not support"):
        _ = await fetch_remote_model_ids(object())


@pytest.mark.asyncio
async def test_fetch_remote_model_ids_timeout() -> None:
    import asyncio

    class Slow:
        async def models(self) -> list[str]:
            await asyncio.sleep(10)
            return ["late"]

    with pytest.raises(TimeoutError):
        _ = await fetch_remote_model_ids(Slow(), timeout_seconds=0.05)


def test_needs_remote_models_for_selection() -> None:
    multi = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"a": ModelConfig(), "b": ModelConfig()},
    )
    single = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={"only": ModelConfig()},
    )
    empty = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://x/v1",
        models={},
    )
    assert needs_remote_models_for_selection(multi, preferred_model="a", interactive=True) is False
    assert needs_remote_models_for_selection(multi, preferred_model=None, interactive=False) is False
    assert needs_remote_models_for_selection(single, preferred_model=None, interactive=True) is False
    assert needs_remote_models_for_selection(multi, preferred_model=None, interactive=True) is True
    assert needs_remote_models_for_selection(empty, preferred_model=None, interactive=True) is True
