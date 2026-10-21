from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config import load

if TYPE_CHECKING:
    from pathlib import Path


def test_notify_section_defaults(tmp_path: Path) -> None:
    """Notifications are on by default: a five-second grace, desktop first."""
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    cfg = load(path).notify_config
    assert cfg.enabled is True
    assert cfg.grace_seconds == 5.0
    assert cfg.min_interval_seconds == 10.0
    assert cfg.channels == ["auto"]
    assert cfg.actions is True
    assert cfg.include_reason is False
    assert cfg.protocol == ""


def test_notify_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[notify]
enabled = false
grace_seconds = 2.5
min_interval_seconds = 30.0
channels = ["terminal"]
actions = false
include_reason = true
protocol = "osc9"
""",
        encoding="utf-8",
    )
    cfg = load(path).notify_config
    assert cfg.enabled is False
    assert cfg.grace_seconds == 2.5
    assert cfg.min_interval_seconds == 30.0
    assert cfg.channels == ["terminal"]
    assert cfg.actions is False
    assert cfg.include_reason is True
    assert cfg.protocol == "osc9"


def test_notify_section_typo_falls_back(tmp_path: Path) -> None:
    """An unknown key or a bad type degrades to the defaults, like ``[mcp]``."""
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[notify]
grace = 1.0
""",
        encoding="utf-8",
    )
    assert load(path).notify_config.grace_seconds == 5.0

    path2 = tmp_path / "d.toml"
    _ = path2.write_text(
        """
[notify]
enabled = "yes"
""",
        encoding="utf-8",
    )
    assert load(path2).notify_config.enabled is True


def test_notify_section_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("[notify]\nenabled = false\n", encoding="utf-8")
    store = load(path)
    assert store.notify_config.enabled is False
    _ = path.write_text("[notify]\nenabled = true\ngrace_seconds = 1.0\n", encoding="utf-8")
    store.reload()
    assert store.notify_config.enabled is True
    assert store.notify_config.grace_seconds == 1.0
