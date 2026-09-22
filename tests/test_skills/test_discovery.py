from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config.models import SkillsConfig
from plyngent.skills import SKILL_FILE, Skill, SkillStore, build_roots, validate_skill_name

if TYPE_CHECKING:
    from pathlib import Path


def _write_skill(root: Path, name: str, *, description: str | None = "A test skill", body: str = "Body.") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    front = "---\n"
    if description is not None:
        front += f"name: {name}\ndescription: {description}\n"
    front += "---\n\n"
    _ = (directory / SKILL_FILE).write_text(front + body + "\n", encoding="utf-8")
    return directory


def test_build_roots_orders_explicit_paths_first(tmp_path: Path) -> None:
    config = SkillsConfig(paths=["~/team-skills", "/srv/skills"])
    roots = build_roots(
        config,
        workspace=tmp_path / "proj",
        user_config_dir=tmp_path / "config",
        home=tmp_path / "home",
    )
    assert [(r.origin, r.scope, str(r.path)) for r in roots] == [
        ("config", "user", str(tmp_path / "home" / "team-skills")),
        ("config", "user", "/srv/skills"),
        ("plyngent", "user", str(tmp_path / "config" / "skills")),
        ("claude", "user", str(tmp_path / "home" / ".claude" / "skills")),
        ("claude", "project", str(tmp_path / "proj" / ".claude" / "skills")),
    ]


def test_build_roots_respects_discover_and_project_switch(tmp_path: Path) -> None:
    config = SkillsConfig(discover=["plyngent"], include_project_roots=False)
    roots = build_roots(config, workspace=tmp_path, user_config_dir=tmp_path / "cfg", home=tmp_path)
    assert [r.origin for r in roots] == ["plyngent"]


def test_store_discovers_skills_and_labels_their_root(tmp_path: Path) -> None:
    user = tmp_path / "config" / "skills"
    _ = _write_skill(user, "pdf-processing", description="Extract text from PDFs")
    store = SkillStore(
        build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path / "config")
    )
    skills = store.skills
    assert [(s.name, s.description, s.root.label) for s in skills] == [
        ("pdf-processing", "Extract text from PDFs", "plyngent-user"),
    ]
    assert skills[0].skill_file.name == SKILL_FILE


def test_store_lists_skills_from_several_roots_sorted(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    _ = _write_skill(config_dir / "skills", "zeta")
    _ = _write_skill(tmp_path / "home" / ".claude" / "skills", "alpha")
    store = SkillStore(
        build_roots(SkillsConfig(), workspace=tmp_path, user_config_dir=config_dir, home=tmp_path / "home")
    )
    assert [s.name for s in store.skills] == ["alpha", "zeta"]
    assert [s.root.origin for s in store.skills] == ["claude", "plyngent"]


def test_explicit_path_wins_and_shadowed_copy_stays_listed(tmp_path: Path) -> None:
    explicit = tmp_path / "team"
    config_dir = tmp_path / "config"
    _ = _write_skill(explicit, "shared", description="Team copy")
    _ = _write_skill(config_dir / "skills", "shared", description="User copy")
    config = SkillsConfig(paths=[str(explicit)], discover=["plyngent"])
    store = SkillStore(build_roots(config, workspace=tmp_path, user_config_dir=config_dir))

    found = store.find("shared")
    assert found is not None
    assert found.description == "Team copy"
    assert [s.description for s in store.shadowed()["shared"]] == ["Team copy", "User copy"]
    assert len(store.all_skills) == 2


def test_find_is_case_insensitive_fallback(tmp_path: Path) -> None:
    _ = _write_skill(tmp_path / "skills", "pdf-processing")
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    assert store.find("PDF-Processing") is not None
    assert store.find("nope") is None


def test_root_itself_may_be_a_skill(tmp_path: Path) -> None:
    single = tmp_path / "my-skill"
    single.mkdir()
    _ = (single / SKILL_FILE).write_text("---\nname: my-skill\ndescription: One-off\n---\n\nBody.\n", encoding="utf-8")
    store = SkillStore(build_roots(SkillsConfig(paths=[str(single)]), workspace=tmp_path, user_config_dir=tmp_path))
    assert [s.name for s in store.skills] == ["my-skill"]


def test_missing_roots_are_not_an_error(tmp_path: Path) -> None:
    store = SkillStore(
        build_roots(SkillsConfig(), workspace=tmp_path, user_config_dir=tmp_path / "nope", home=tmp_path)
    )
    assert store.skills == []
    assert store.shadowed() == {}


def test_description_falls_back_to_the_first_paragraph(tmp_path: Path) -> None:
    directory = tmp_path / "skills" / "no-desc"
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text(
        "# Heading\n\nFirst paragraph of the skill.\n\nSecond paragraph.\n",
        encoding="utf-8",
    )
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    skill = store.skills[0]
    assert skill.description == "First paragraph of the skill."
    assert skill.declared_name is None
    assert any("no frontmatter description" in issue for issue in skill.issues)


def test_unusable_declared_name_falls_back_to_the_directory(tmp_path: Path) -> None:
    directory = tmp_path / "skills" / "dir-name"
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text(
        '---\nname: "PDF Skills"\ndescription: Mixed case\n---\n\nBody.\n',
        encoding="utf-8",
    )
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    skill = store.skills[0]
    assert skill.name == "dir-name"
    assert skill.declared_name == "PDF Skills"
    assert any("not usable" in issue for issue in skill.issues)


def test_broken_frontmatter_still_yields_a_listable_skill(tmp_path: Path) -> None:
    directory = tmp_path / "skills" / "broken"
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text("---\nname: broken\nbroken line\n---\n\nBody text.\n", encoding="utf-8")
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    skill = store.skills[0]
    assert skill.name == "broken"
    assert skill.description  # falls back to the body
    assert any("frontmatter not parsed" in issue for issue in skill.issues)


def test_cache_picks_up_an_edit_and_reload_forces_a_rescan(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    directory = _write_skill(root, "cached", description="Before")
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    assert store.skills[0].description == "Before"

    _ = (directory / SKILL_FILE).write_text(
        "---\nname: cached\ndescription: After\n---\n\nBody.\n",
        encoding="utf-8",
    )
    assert store.skills[0].description == "After"

    _ = _write_skill(root, "added")
    store.reload()
    assert [s.name for s in store.skills] == ["added", "cached"]


def test_default_write_root_prefers_our_own_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    store = SkillStore(build_roots(SkillsConfig(), workspace=tmp_path, user_config_dir=config_dir, home=tmp_path))
    root = store.default_write_root()
    assert root is not None
    assert root.origin == "plyngent"
    assert root.path == config_dir / "skills"

    explicit = SkillStore(
        build_roots(SkillsConfig(paths=["/srv/skills"], discover=[]), workspace=tmp_path, user_config_dir=config_dir)
    )
    fallback = explicit.default_write_root()
    assert fallback is not None and fallback.origin == "config"
    assert SkillStore([]).default_write_root() is None


def test_validate_skill_name() -> None:
    assert validate_skill_name("pdf-processing") is None
    assert validate_skill_name("a1") is None
    assert validate_skill_name("") is not None
    assert validate_skill_name("PDF") is not None
    assert validate_skill_name("with_underscore") is not None
    assert validate_skill_name("x" * 65) is not None


def test_skill_is_a_frozen_record(tmp_path: Path) -> None:
    directory = _write_skill(tmp_path / "skills", "frozen")
    store = SkillStore(build_roots(SkillsConfig(discover=["plyngent"]), workspace=tmp_path, user_config_dir=tmp_path))
    skill = store.skills[0]
    assert isinstance(skill, Skill)
    assert skill.path == directory
    assert skill.metadata["name"] == "frozen"
