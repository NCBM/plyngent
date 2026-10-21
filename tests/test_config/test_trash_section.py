from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config import load

if TYPE_CHECKING:
    from pathlib import Path


def test_trash_section_defaults(tmp_path: Path) -> None:
    """Deletes go to the system trash unless the section says otherwise."""
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    assert load(path).trash_config.enabled is True


def test_trash_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("[trash]\nenabled = false\n", encoding="utf-8")
    assert load(path).trash_config.enabled is False


def test_trash_section_typo_falls_back(tmp_path: Path) -> None:
    """An unknown key or a bad type degrades to the defaults, like ``[mcp]``."""
    path = tmp_path / "c.toml"
    _ = path.write_text("[trash]\nenable = false\n", encoding="utf-8")
    assert load(path).trash_config.enabled is True

    path2 = tmp_path / "d.toml"
    _ = path2.write_text('[trash]\nenabled = "no"\n', encoding="utf-8")
    assert load(path2).trash_config.enabled is True


def test_trash_section_survives_reload(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("[trash]\nenabled = false\n", encoding="utf-8")
    store = load(path)
    assert store.trash_config.enabled is False
    _ = path.write_text("[trash]\nenabled = true\n", encoding="utf-8")
    store.reload()
    assert store.trash_config.enabled is True
