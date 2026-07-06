import shutil
from collections.abc import Mapping
from pathlib import Path

import pytest

import plyngent
from plyngent.config import (
    AnthropicProvider,
    ConfigFormatError,
    DeepseekProvider,
    OpenAICompatibleProvider,
    OpenAIProvider,
)


@pytest.fixture
def default_config_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plyngent.config, "default_config_source", Path(__file__).parent / "plyngent-valid.toml")


def test_read_default_config(default_config_source: None) -> None:
    _ = default_config_source
    config = plyngent.config.load()
    providers = config.providers
    assert isinstance(providers, Mapping)
    assert isinstance(providers["test1"], OpenAIProvider)
    assert isinstance(providers["test2"], OpenAICompatibleProvider)
    assert isinstance(providers["test3"], AnthropicProvider)
    assert isinstance(providers["foo1"], DeepseekProvider)


def test_read_valid_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-valid.toml")
    providers = config.providers
    assert isinstance(providers, Mapping)
    assert isinstance(providers["test1"], OpenAIProvider)
    assert isinstance(providers["test2"], OpenAICompatibleProvider)
    assert isinstance(providers["test3"], AnthropicProvider)
    assert isinstance(providers["foo1"], DeepseekProvider)


def test_read_empty_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-empty.toml")
    assert isinstance(config.providers, Mapping)
    assert not config.providers


def test_read_bad_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-bad.toml")
    assert isinstance(config.providers, Mapping)
    assert not config.providers
    assert isinstance(config.bad_providers, Mapping)


def test_read_invalid_config() -> None:
    with pytest.raises(ConfigFormatError):
        _ = plyngent.config.load(Path(__file__).parent / "plyngent-invalid.toml")


def test_write_new_config() -> None:
    file = Path(__file__).parent / "plyngent-edit-1.toml"
    file.unlink(missing_ok=True)
    config = plyngent.config.load(file)
    assert isinstance(config.providers, Mapping)
    config.providers = {
        "foo1": OpenAIProvider(access_key_or_token="sk-00301212"),
        "foo2": DeepseekProvider(access_key_or_token="sk-00301212"),
    }
    assert isinstance(config.providers, Mapping)
    config.write()
    assert isinstance(config.providers["foo1"], OpenAIProvider)
    assert config.providers["foo1"].access_key_or_token == "sk-00301212"
    assert isinstance(config.providers["foo2"], DeepseekProvider)
    assert config.providers["foo2"].access_key_or_token == "sk-00301212"
    config.reload()
    assert isinstance(config.providers["foo1"], OpenAIProvider)
    assert config.providers["foo1"].access_key_or_token == "sk-00301212"
    assert isinstance(config.providers["foo2"], DeepseekProvider)
    assert config.providers["foo2"].access_key_or_token == "sk-00301212"


def test_update_config() -> None:
    file = Path(__file__).parent / "plyngent-edit-2.toml"
    _ = shutil.copy(Path(__file__).parent / "plyngent-valid.toml", file)
    config = plyngent.config.load(file)
    assert isinstance(config.providers, Mapping)
    config.providers = config.providers | {"foo2": DeepseekProvider(access_key_or_token="sk-00301212")}
    assert isinstance(config.providers, Mapping)
    config.write()
    assert isinstance(config.providers["foo1"], DeepseekProvider)
    assert config.providers["foo1"].access_key_or_token == "sk-1145141919810"
    assert isinstance(config.providers["foo2"], DeepseekProvider)
    assert config.providers["foo2"].access_key_or_token == "sk-00301212"
    config.reload()
    assert isinstance(config.providers["foo1"], DeepseekProvider)
    assert config.providers["foo1"].access_key_or_token == "sk-1145141919810"
    assert isinstance(config.providers["foo2"], DeepseekProvider)
    assert config.providers["foo2"].access_key_or_token == "sk-00301212"
