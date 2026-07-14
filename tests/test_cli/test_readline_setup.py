from __future__ import annotations

import tomlkit

from plyngent.cli.readline_setup import (
    SLASH_COMMANDS,
    build_completer,
    filter_prefix,
    history_path,
)
from plyngent.cli.state import ReplState
from plyngent.config.models import ModelConfig, OpenAICompatibleProvider, OpenAIProvider
from plyngent.config.store import ConfigStore


def _minimal_state(tmp_path: object) -> ReplState:
    from pathlib import Path
    from unittest.mock import MagicMock

    assert isinstance(tmp_path, Path)
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk",
        url="https://example.com/v1",
        models={"alpha": ModelConfig(), "beta": ModelConfig()},
    )
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
    config.providers = {
        "local": OpenAIProvider(access_key_or_token="sk"),
        "remote": provider,
    }
    # Avoid real client/network: build ReplState pieces manually via object.__new__
    state = object.__new__(ReplState)
    state.config = config
    state.provider = provider
    state.provider_name = "remote"
    state.model = "alpha"
    state.tools_enabled = True
    state.memory = MagicMock()
    state.workspace = tmp_path
    state.session_id = None
    state.client = MagicMock()
    state.agent = MagicMock()
    return state


def test_filter_prefix() -> None:
    assert filter_prefix("/he", ["/help", "/quit"]) == ["/help"]
    assert filter_prefix("", ["a", "b"]) == ["a", "b"]


def test_history_path_under_user_data() -> None:
    path = history_path()
    assert path.name == "repl_history"
    assert "plyngent" in str(path)


def test_completer_commands(tmp_path: object, monkeypatch: object) -> None:
    import readline
    from pathlib import Path

    import pytest

    assert isinstance(tmp_path, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    state = _minimal_state(tmp_path)
    completer = build_completer(state)
    monkeypatch.setattr(readline, "get_line_buffer", lambda: "/")
    monkeypatch.setattr(readline, "get_begidx", lambda: 0)

    found: list[str] = []
    index = 0
    while True:
        item = completer("/", index)
        if item is None:
            break
        found.append(item)
        index += 1
    assert "/help" in found
    assert set(found) <= set(SLASH_COMMANDS)


def test_completer_provider_args(tmp_path: object, monkeypatch: object) -> None:
    import readline
    from pathlib import Path

    import pytest

    assert isinstance(tmp_path, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)

    state = _minimal_state(tmp_path)
    completer = build_completer(state)
    monkeypatch.setattr(readline, "get_line_buffer", lambda: "/provider ")
    monkeypatch.setattr(readline, "get_begidx", lambda: len("/provider "))

    first = completer("r", 0)
    assert first == "remote"
