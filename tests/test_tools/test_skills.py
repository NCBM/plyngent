from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from plyngent.config.models import SkillsConfig
from plyngent.skills import SKILL_FILE, SkillStore, build_roots
from plyngent.tools.context import InstanceState, SessionState, bind_instance, bind_session
from plyngent.tools.skills import (
    SKILL_TOOL_NAMES,
    SkillWriteDecision,
    SkillWritePolicy,
    format_skill_table,
    set_skill_write_policy,
    skill_create,
    skill_edit,
    skill_list,
    skill_read,
    skill_search,
)
from plyngent.tools.workspace import set_path_denylist, set_static_read_roots, set_workspace_root

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
def _bound(
    tmp_path: Path,
    *roots: Path,
    skills: SkillStore | None = None,
    session: bool = True,
    policy: SkillWritePolicy | None = None,
) -> Generator[InstanceState]:
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
    session_state = SessionState(session_id=1) if session else None
    with bind_instance(instance), bind_session(session_state):
        _ = set_workspace_root(workspace)
        _ = set_static_read_roots(roots)
        if policy is not None:
            set_skill_write_policy(policy, instance=instance)
        yield instance


def _write_root(tmp_path: Path) -> Path:
    """An empty user-level skills directory to create skills in."""
    root = tmp_path / "ours"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allow(*_args: object) -> SkillWriteDecision:
    return SkillWriteDecision(allow=True)


def _deny(*_args: object) -> bool:
    return False


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


async def test_skill_search_inside_the_workspace_rebases_the_hit(tmp_path: Path) -> None:
    """A project-scope skill root lives inside the workspace: one path, once.

    The search reports in-workspace files relative to the workspace root and outside
    ones absolutely, so the skill name must never be glued onto a path that
    already carries the root.
    """
    root = tmp_path / "project" / ".claude" / "skills"
    _ = _write_skill(root, "alpha", body="Use pdftotext here.")
    with _bound(tmp_path, root):
        out = await skill_search.handler("pdftotext")
    assert out.count("alpha/SKILL.md:") == 1
    assert "project/.claude/skills/alpha" not in out
    assert "alpha/alpha/SKILL.md" not in out


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
    expected = ("skill_list", "skill_read", "skill_search", "skill_create", "skill_edit")
    for name in expected:
        assert name in SKILL_TOOL_NAMES
    assert len(SKILL_TOOL_NAMES) == len(expected)


async def test_skill_create_writes_skill_md_and_bundled_files(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        out = await skill_create.handler(
            "acme-deploy",
            "Deploy the acme stack",
            "# Steps\n\n1. Build.\n",
            files={"scripts/deploy.sh": "#!/bin/sh\necho deploy\n"},
        )
        assert out.startswith("created skill 'acme-deploy' at ")
        assert (root / "acme-deploy" / SKILL_FILE).is_file()
        assert (root / "acme-deploy" / "scripts" / "deploy.sh").read_text(encoding="utf-8").startswith("#!/bin/sh")
        # The new skill is visible right away, without a restart.
        assert "acme-deploy  [config-user]" in await skill_list.handler()
        read = await skill_read.handler("acme-deploy")
    assert "description: Deploy the acme stack" in read
    assert "1. Build." in read


async def test_skill_create_quotes_a_frontmatter_description_that_needs_it(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        _ = await skill_create.handler("quoted", "Use: colons, {braces}", "Body.")
        skill = SkillStore(
            build_roots(SkillsConfig(paths=[str(root)], discover=[]), workspace=tmp_path, user_config_dir=tmp_path)
        ).skills[0]
    assert skill.description == "Use: colons, {braces}"


async def test_skill_create_rejects_a_bad_name_and_a_taken_one(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    _ = _write_skill(root, "taken")
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        assert (await skill_create.handler("Not-A-Name", "d", "b")).startswith("error: skill name")
        assert "already exists" in await skill_create.handler("taken", "d", "b")
        assert (await skill_create.handler("ok-name", "d", "b")).startswith("created skill")


async def test_skill_create_rejects_a_bundled_file_outside_the_skill(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        out = await skill_create.handler("acme", "d", "b", files={"../escape.sh": "x"})
    assert out == "error: bundled file '../escape.sh' is outside skill 'acme'"
    assert not (tmp_path / "escape.sh").exists()


async def test_skill_write_needs_a_grant(tmp_path: Path) -> None:
    """Without a hook the write is denied, and the message says how to open it."""
    root = _write_root(tmp_path)
    with _bound(tmp_path, root):
        out = await skill_create.handler("acme", "d", "b")
    assert out == (
        "error: writing skills needs a grant (no confirm hook is installed; denied); "
        "set [skills].allow_write = true to allow skill writes"
    )
    assert not (root / "acme").exists()


async def test_skill_write_prompt_denial_is_reported(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    _ = _write_skill(root, "acme", body="Before.")
    with _bound(tmp_path, root, policy=SkillWritePolicy(confirm=_deny, timeout=3.0)):
        out = await skill_edit.handler("acme", old_string="Before.", new_string="After.")
    assert out.startswith("error: skill write to ")
    assert out.endswith("denied (declined or timed out after 3s)")
    assert "After." not in (root / "acme" / SKILL_FILE).read_text(encoding="utf-8")


async def test_skill_write_confirm_sees_the_action_and_the_skill(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    seen: list[tuple[Path, str, str, float]] = []

    def hook(target: Path, action: str, detail: str, timeout: float) -> bool:
        seen.append((target, action, detail, timeout))
        return True

    with _bound(tmp_path, root, policy=SkillWritePolicy(confirm=hook, timeout=5.0)):
        _ = await skill_create.handler("acme", "d", "b")
    assert seen == [(root / "acme", "create", f"create skill 'acme' at {root / 'acme'} (SKILL.md)", 5.0)]


async def test_skill_write_session_grant_is_asked_once(tmp_path: Path) -> None:
    """A 'session' answer covers the same skill for the rest of the chat."""
    root = _write_root(tmp_path)
    calls: list[Path] = []

    def hook(target: Path, action: str, detail: str, timeout: float) -> SkillWriteDecision:
        calls.append(target)
        return SkillWriteDecision(allow=True, session=True)

    with _bound(tmp_path, root, policy=SkillWritePolicy(confirm=hook)):
        _ = await skill_create.handler("acme", "d", "b")
        _ = await skill_edit.handler("acme", append="\nMore.\n")
        _ = await skill_edit.handler("acme", file="notes.md", new_string="note\n")
    # One prompt for the skill directory, covering SKILL.md and later files.
    assert calls == [root / "acme"]
    assert (root / "acme" / "notes.md").read_text(encoding="utf-8") == "note\n"


async def test_skill_write_denylist_wins_over_a_grant(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        set_path_denylist(["acme"])
        try:
            out = await skill_create.handler("acme", "d", "b")
        finally:
            set_path_denylist(())
    assert out.startswith("error: path denied by policy (matched 'acme')")
    assert not (root / "acme").exists()


async def test_skill_edit_forms(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    _ = _write_skill(root, "acme", body="Line one.\nLine two.")
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        out = await skill_edit.handler("acme", old_string="Line two.", new_string="Line two, edited.")
        assert out.startswith("updated skill 'acme' (SKILL.md: replaced 1 of 1 occurrence(s))")

        out = await skill_edit.handler("acme", append="\nAppended tail.")
        assert "appended 15 characters" in out

        out = await skill_edit.handler("acme", file="scripts/run.sh", new_string="#!/bin/sh\n")
        assert "rewrote 10 characters" in out
        assert (root / "acme" / "scripts" / "run.sh").is_file()
    text = (root / "acme" / SKILL_FILE).read_text(encoding="utf-8")
    assert "Line two, edited." in text
    assert text.endswith("Appended tail.")


async def test_skill_edit_replace_all_and_missing_match(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    directory = _write_skill(root, "acme", body="hit\nhit\nhit")
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        out = await skill_edit.handler("acme", old_string="hit", new_string="done", replace_all=True)
        assert "replaced 3 of 3 occurrence(s)" in out
        missing = await skill_edit.handler("acme", old_string="nope", new_string="x")
    assert missing == "error: old_string not found in SKILL.md of skill 'acme'"
    assert (directory / SKILL_FILE).read_text(encoding="utf-8").count("done") == 3


async def test_skill_edit_rejects_conflicting_forms(tmp_path: Path) -> None:
    root = _write_root(tmp_path)
    _ = _write_skill(root, "acme")
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        assert (await skill_edit.handler("acme")).startswith("error: nothing to write")
        assert (await skill_edit.handler("acme", old_string="a", new_string="b", append="c")).startswith(
            "error: use either"
        )
        assert (await skill_edit.handler("acme", file="nope.md", append="x")) == (
            "error: no such file in skill 'acme': nope.md"
        )
        assert (await skill_edit.handler("acme", file="../x.md", new_string="x")).startswith("error: file")


async def test_skill_edit_notes_a_rewritten_frontmatter(tmp_path: Path) -> None:
    """A SKILL.md that loses its description (or breaks) is reported, not repaired."""
    root = _write_root(tmp_path)
    _ = _write_skill(root, "acme")
    with _bound(tmp_path, root, policy=SkillWritePolicy(allow_all=True)):
        dropped = await skill_edit.handler("acme", new_string="Only a body, no frontmatter.\n")
        broken = await skill_edit.handler("acme", new_string="---\nname: acme\nbroken line\n---\n\nBody.\n")
    assert dropped.endswith("notes: SKILL.md has no frontmatter description; the first paragraph will be used")
    assert "notes: SKILL.md frontmatter not parsed" in broken


async def test_skill_tools_reject_writes_when_no_session_is_bound(tmp_path: Path) -> None:
    """A host without session state still denies instead of crashing on the grant."""
    root = _write_root(tmp_path)
    _ = _write_skill(root, "acme", body="Before.")
    with _bound(tmp_path, root, session=False, policy=SkillWritePolicy(confirm=_allow)):
        out = await skill_edit.handler("acme", old_string="Before.", new_string="After.")
    assert out.startswith("updated skill 'acme'")
