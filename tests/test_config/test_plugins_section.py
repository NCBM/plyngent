"""[plugins] section mutators and parse."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tomlkit

from plyngent.config import load
from plyngent.config.store import ConfigStore

if TYPE_CHECKING:
    from pathlib import Path


def test_plugins_defaults_and_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    store = load(path)
    assert store.plugins_config.enable == []
    assert store.plugins_config.disable == []

    path.write_text(
        """
[plugins]
enable = ["acme", "*"]
disable = ["legacy"]
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.plugins_config.enable == ["acme", "*"]
    assert store.plugins_config.disable == ["legacy"]


def test_enable_disable_undeny_write(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    store = ConfigStore(path=path, document=tomlkit.document())
    _ = store.enable_plugin("acme")
    assert store.plugins_config.enable == ["acme"]
    store.write()
    reloaded = load(path)
    assert reloaded.plugins_config.enable == ["acme"]

    _ = store.disable_plugin("acme")
    assert "acme" not in store.plugins_config.enable
    assert store.plugins_config.disable == ["acme"]
    store.write()
    reloaded = load(path)
    assert reloaded.plugins_config.disable == ["acme"]

    _ = store.undeny_plugin("acme")
    assert store.plugins_config.disable == []
    _ = store.enable_plugin("*")
    assert store.plugins_config.enable == ["*"]
    _ = store.enable_plugin("other")
    # Already * — stay *
    assert store.plugins_config.enable == ["*"]
    _ = store.clear_plugins()
    assert store.plugins_config.enable == []
    assert store.plugins_config.disable == []
