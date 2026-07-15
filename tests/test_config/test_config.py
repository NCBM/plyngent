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
    db = config.database
    assert db["implementation"] == "sqlite"
    assert db["url"] == ":memory:"
    assert db["username"] is None
    assert db["password"] is None


def test_read_valid_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-valid.toml")
    providers = config.providers
    assert isinstance(providers, Mapping)
    assert isinstance(providers["test1"], OpenAIProvider)
    assert isinstance(providers["test2"], OpenAICompatibleProvider)
    assert isinstance(providers["test3"], AnthropicProvider)
    assert isinstance(providers["foo1"], DeepseekProvider)
    # TOML omitted models → DeepSeek defaults.
    assert set(providers["foo1"].models) == {"deepseek-v4-flash", "deepseek-v4-pro"}
    assert providers["foo1"].models["deepseek-v4-flash"].text is True
    db = config.database
    assert db["implementation"] == "sqlite"
    assert db["url"] == ":memory:"
    assert db["username"] is None
    assert db["password"] is None


def test_deepseek_default_models_on_construct() -> None:
    provider = DeepseekProvider(access_key_or_token="sk-test")
    assert set(provider.models) == {"deepseek-v4-flash", "deepseek-v4-pro"}


def test_openai_default_models_on_construct() -> None:
    provider = OpenAIProvider(access_key_or_token="sk-test")
    assert set(provider.models) == {"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}


def test_openai_omitted_preset_and_models_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "openai-defaults.toml"
    _ = path.write_text(
        """
[providers.oai]
access_key_or_token = "sk-test"
""",
        encoding="utf-8",
    )
    config = plyngent.config.load(path)
    assert "oai" in config.providers
    provider = config.providers["oai"]
    assert isinstance(provider, OpenAIProvider)
    assert set(provider.models) == {"gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}


def test_deepseek_explicit_models_override_defaults() -> None:
    from plyngent.config import ModelConfig

    provider = DeepseekProvider(
        access_key_or_token="sk-test",
        models={"custom-only": ModelConfig(text=True)},
    )
    assert set(provider.models) == {"custom-only"}


def test_read_empty_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-empty.toml")
    assert isinstance(config.providers, Mapping)
    assert not config.providers


def test_read_bad_config() -> None:
    config = plyngent.config.load(Path(__file__).parent / "plyngent-bad.toml")
    assert isinstance(config.providers, Mapping)
    assert not config.providers
    assert isinstance(config.bad_providers, Mapping)


def test_provider_with_empty_models_is_recoverable(tmp_path: Path) -> None:
    path = tmp_path / "empty-models.toml"
    _ = path.write_text(
        """
[providers.hollow]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk-test"
models = {}
""",
        encoding="utf-8",
    )
    config = plyngent.config.load(path)
    assert "hollow" not in config.providers
    assert "hollow" not in config.bad_providers
    assert "hollow" in config.recoverable_providers
    promoted = config.promote_provider("hollow", ["m1", "m2"])
    assert "hollow" in config.providers
    assert "hollow" not in config.recoverable_providers
    assert set(promoted.models) == {"m1", "m2"}


def test_promote_provider_requires_ids(tmp_path: Path) -> None:
    path = tmp_path / "empty-models.toml"
    _ = path.write_text(
        """
[providers.hollow]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk-test"
models = {}
""",
        encoding="utf-8",
    )
    config = plyngent.config.load(path)
    with pytest.raises(ValueError, match="no model ids"):
        _ = config.promote_provider("hollow", [])


def test_read_invalid_config() -> None:
    with pytest.raises(ConfigFormatError):
        _ = plyngent.config.load(Path(__file__).parent / "plyngent-invalid.toml")


def test_write_new_config() -> None:
    file = Path(__file__).parent / "plyngent-edit-1.toml"
    file.unlink(missing_ok=True)
    config = plyngent.config.load(file)
    assert isinstance(config.providers, Mapping)
    from plyngent.config import ModelConfig

    config.providers = {
        "foo1": OpenAIProvider(
            access_key_or_token="sk-00301212",
            models={"gpt-test": ModelConfig()},
        ),
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
