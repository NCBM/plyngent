from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config import load

if TYPE_CHECKING:
    from pathlib import Path


def test_skills_section_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    store = load(path)
    assert store.skills_config.enabled is True
    assert store.skills_config.paths == []
    assert store.skills_config.discover == ["plyngent", "claude"]
    assert store.skills_config.include_project_roots is True
    assert store.skills_config.inject_catalog is True
    assert store.skills_config.max_catalog_skills == 50
    assert store.skills_config.allow_write is False


def test_skills_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[skills]
enabled = false
paths = ["~/team-skills", "/srv/skills"]
discover = ["plyngent"]
include_project_roots = false
inject_catalog = false
max_catalog_skills = 5
allow_write = true
""",
        encoding="utf-8",
    )
    cfg = load(path).skills_config
    assert cfg.enabled is False
    assert cfg.paths == ["~/team-skills", "/srv/skills"]
    assert cfg.discover == ["plyngent"]
    assert cfg.include_project_roots is False
    assert cfg.inject_catalog is False
    assert cfg.max_catalog_skills == 5
    assert cfg.allow_write is True


def test_skills_section_invalid_falls_back(tmp_path: Path) -> None:
    """An unknown discovery source or a bad type degrades to the defaults."""
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[skills]
discover = ["codex"]
""",
        encoding="utf-8",
    )
    assert load(path).skills_config.discover == ["plyngent", "claude"]

    path2 = tmp_path / "d.toml"
    _ = path2.write_text(
        """
[skills]
enabled = "yes"
""",
        encoding="utf-8",
    )
    assert load(path2).skills_config.enabled is True
