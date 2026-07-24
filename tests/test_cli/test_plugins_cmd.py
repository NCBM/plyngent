"""CLI ``plyngent plugins`` and slash ``/plugins``."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import tomlkit
from click.testing import CliRunner

from plyngent.cli.app import main
from plyngent.cli.slash import handle_slash
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.memory import MemoryStore
from plyngent.tools.plugins import DiscoveredPlugin, PluginStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


@pytest.fixture
async def state(tmp_path: Path) -> AsyncIterator[ReplState]:
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=tomlkit.document())
    config.providers = {"local": provider}
    st = ReplState(
        config=config,
        memory=memory,
        workspace=tmp_path,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=False,
    )
    # ReplState builds a real client; slash plugin tests only need config + agent.
    await st.new_session("t")
    yield st
    await memory.close()


def _fake_statuses(*, enable: list[str], disable: list[str]) -> list[PluginStatus]:
    del enable, disable
    return [
        PluginStatus(
            plugin=DiscoveredPlugin(id="acme", value="acme_pkg:load", package="acme-pkg", version="1.0"),
            enabled=True,
            disabled=False,
        ),
        PluginStatus(
            plugin=DiscoveredPlugin(id="other", value="other:load", package=None, version=None),
            enabled=False,
            disabled=True,
        ),
    ]


def test_plugins_list_cmd(tmp_path: Path) -> None:
    config = tmp_path / "plyngent.toml"
    _ = config.write_text(
        """
[plugins]
enable = ["acme"]
disable = ["other"]
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    with patch("plyngent.tools.plugins.list_plugin_statuses", side_effect=_fake_statuses):
        result = runner.invoke(main, ["plugins", "list", "--config", str(config)])
    assert result.exit_code == 0
    assert "acme" in result.output
    assert "enabled" in result.output
    assert "other" in result.output
    assert "disabled" in result.output


def test_plugins_enable_disable_write(tmp_path: Path) -> None:
    config = tmp_path / "plyngent.toml"
    _ = config.write_text("", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["plugins", "enable", "acme", "--config", str(config)])
    assert result.exit_code == 0
    text = config.read_text(encoding="utf-8")
    assert "acme" in text
    result = runner.invoke(main, ["plugins", "disable", "acme", "--config", str(config)])
    assert result.exit_code == 0
    text = config.read_text(encoding="utf-8")
    assert "disable" in text


async def test_slash_plugins_enable_reload(state: ReplState) -> None:
    with patch("plyngent.tools.plugins.list_plugin_statuses", return_value=[]):
        assert await handle_slash(state, "/plugins list") is True
    assert await handle_slash(state, "/plugins enable acme") is True
    assert "acme" in state.config.plugins_config.enable
    assert await handle_slash(state, "/plugins disable acme") is True
    assert "acme" in state.config.plugins_config.disable
    assert await handle_slash(state, "/plugins undeny acme") is True
    assert "acme" not in state.config.plugins_config.disable
    assert await handle_slash(state, "/plugins reload") is True
