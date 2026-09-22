from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from plyngent.config.models import SkillsConfig
from plyngent.skills import SKILL_FILE, SkillStore, build_roots
from plyngent.tools.context import InstanceState, bind_instance
from plyngent.tools.skills import SKILL_TOOL_NAMES, format_skill_table, skill_list, skill_read, skill_search
from plyngent.tools.workspace import set_static_read_roots, set_workspace_root

if TYPE_CHECKING:
    from collections.abc import Generator
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


@contextmanager
def _bound(tmp_path: Path, *roots: Path, skills: SkillStore | None = None) -> Generator[InstanceState]:
    """Bind an instance whose skill store scans *roots*, like the CLI does."""
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    store = (
        skills
        if skills is not None
        else SkillStore(
            build_roots(
                SkillsConfig(paths=[str(r) for r in roots], discover=[]), workspace=workspace, user_config_dir=tmp_path
            )
        )
    )
    instance = InstanceState(workspace_root=workspace, skills=store)
    with bind_instance(instance):
        _ = set_workspace_root(workspace)
        _ = set_static_read_roots(roots)
        yield instance


async def test_skill_list_without_a_store_returns_an_error(tmp_path: Path) -> None:
    with _bound(tmp_path, skills=None) as instance:
        instance.skills = None
        assert await skill_list.handler() == "error: skills are not available in this session"


async def test_skill_list_reports_skills_roots_and_absent_dirs(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme-deploy", description="Deploy the acme stack")
    missing = tmp_path / "missing"
    with _bound(tmp_path, root, missing):
        out = await skill_list.handler()
    assert "skills: 1" in out
    assert "acme-deploy  [config-user]" in out
    assert "Deploy the acme stack" in out
    assert str(root / "acme-deploy") in out
    assert "roots without a directory" in out
    assert str(missing) in out


async def test_skill_list_marks_shadowed_copies(tmp_path: Path) -> None:
    high = tmp_path / "team"
    low = tmp_path / "personal"
    _ = _write_skill(high, "shared", description="Team copy")
    _ = _write_skill(low, "shared", description="Personal copy")
    with _bound(tmp_path, high, low):
        out = await skill_list.handler()
    assert "Team copy" in out
    assert "Personal copy" not in out
    assert "shadowed: shared — config-user wins (also config-user)" in out


async def test_skill_list_shows_notes(tmp_path: Path) -> None:
    root = tmp_path / "team"
    directory = root / "no-desc"
    directory.mkdir(parents=True)
    _ = (directory / SKILL_FILE).write_text("# Heading\n\nBody paragraph.\n", encoding="utf-8")
    with _bound(tmp_path, root):
        out = await skill_list.handler()
    assert "notes: no-desc: no frontmatter description; using the first paragraph" in out


async def test_skill_read_defaults_to_skill_md(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme", body="Step one.")
    with _bound(tmp_path, root):
        out = await skill_read.handler("acme")
    assert out.startswith("L1-")
    assert "Step one." in out


async def test_skill_read_bundled_file_with_line_numbers(tmp_path: Path) -> None:
    root = tmp_path / "team"
    directory = _write_skill(root, "acme")
    (directory / "scripts").mkdir()
    _ = (directory / "scripts" / "run.sh").write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    with _bound(tmp_path, root):
        out = await skill_read.handler("acme", "scripts/run.sh", with_lineno=True)
    assert "     1|#!/bin/sh" in out


async def test_skill_read_refuses_to_escape_the_skill(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme")
    _ = (tmp_path / "outside.txt").write_text("not yours\n", encoding="utf-8")
    with _bound(tmp_path, root):
        assert await skill_read.handler("acme", "../outside.txt") == (
            "error: file '../outside.txt' is outside skill 'acme'"
        )
        assert await skill_read.handler("acme", "/etc/hostname") == (
            "error: file '/etc/hostname' is outside skill 'acme'"
        )


async def test_skill_read_unknown_skill_and_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme")
    with _bound(tmp_path, root):
        assert (
            await skill_read.handler("nope")
            == "error: unknown skill 'nope'; call skill_list to see the discovered skills"
        )
        assert await skill_read.handler("acme", "missing.md") == "error: no such file in skill 'acme': missing.md"


async def test_skill_read_truncates_with_a_continuation_token(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme", body="x" * 500)
    with _bound(tmp_path, root):
        out = await skill_read.handler("acme", max_chars=120)
    assert "truncate_token=" in out


async def test_skill_search_attributes_hits_to_their_skill(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "alpha", body="Use pdftotext for extraction.")
    beta = _write_skill(root, "beta", body="No match in the body.")
    _ = (beta / "notes.md").write_text("pdftotext again\n", encoding="utf-8")
    with _bound(tmp_path, root):
        out = await skill_search.handler("pdftotext")
    assert "alpha/SKILL.md:" in out
    assert "beta/notes.md:" in out


async def test_skill_search_single_skill_and_errors(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "alpha", body="Use pdftotext.")
    _ = _write_skill(root, "beta", body="Nothing to see.")
    with _bound(tmp_path, root):
        assert await skill_search.handler("pdftotext", skill="beta") == "(no matches)"
        assert (await skill_search.handler("pdftotext", skill="nope")).startswith("error: unknown skill")
        assert (await skill_search.handler("[")).startswith("error: invalid regex")


async def test_skill_search_without_skills(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with _bound(tmp_path, empty):
        assert await skill_search.handler("anything") == "(no skills to search)"


async def test_skill_search_respects_max_matches(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "alpha", body="hit\nhit\nhit\nhit\n")
    with _bound(tmp_path, root):
        out = await skill_search.handler("hit", max_matches=2)
    assert out.count("alpha/SKILL.md:") == 2
    assert out.endswith("...[truncated at 2 matches]")


def test_format_skill_table_lists_roots_in_priority_order(tmp_path: Path) -> None:
    root = tmp_path / "team"
    _ = _write_skill(root, "acme")
    store = SkillStore(
        build_roots(SkillsConfig(paths=[str(root)], discover=[]), workspace=tmp_path, user_config_dir=tmp_path)
    )
    table = format_skill_table(store)
    assert "skills: 1" in table
    assert "acme  [config-user]" in table


def test_skill_tool_names_cover_the_group() -> None:
    expected = ("skill_list", "skill_read", "skill_search")
    for name in expected:
        assert name in SKILL_TOOL_NAMES
    assert len(SKILL_TOOL_NAMES) == len(expected)
