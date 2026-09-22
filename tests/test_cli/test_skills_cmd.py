"""CLI surface for skills: ``/skills``, the system-prompt catalog, and writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import tomlkit

from plyngent.cli.slash import handle_slash
from plyngent.cli.state import ReplState
from plyngent.config.models import DatabaseConfig, OpenAIProvider
from plyngent.config.store import ConfigStore
from plyngent.memory import MemoryStore
from plyngent.prompting import NonInteractiveBackend, temporary_backend
from plyngent.skills import SKILL_FILE
from tests.test_prompting import ScriptedBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path


def _write_skill(root: Path, name: str, *, description: str = "A test skill", body: str = "Body.") -> Path:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    _ = (directory / SKILL_FILE).write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return directory


def _config_doc(tmp_path: Path, *, paths: list[Path] | None = None, extra: str = "") -> tomlkit.TOMLDocument:
    lines = ["[skills]", "discover = []"]
    if paths:
        rendered = ", ".join(f'"{path}"' for path in paths)
        lines.append(f"paths = [{rendered}]")
    if extra:
        lines.append(extra)
    return tomlkit.parse("\n".join(lines) + "\n")


async def _state(
    tmp_path: Path,
    *,
    skills_dir: Path | None = None,
    tools_enabled: bool = False,
    extra: str = "",
) -> ReplState:
    memory = await MemoryStore.open(DatabaseConfig())
    provider = OpenAIProvider(access_key_or_token="sk-test")
    document = _config_doc(tmp_path, paths=[skills_dir] if skills_dir is not None else None, extra=extra)
    config = ConfigStore(path=tmp_path / "plyngent.toml", document=document)
    config.providers = {"local": provider}
    # The skills root sits outside the workspace: reading it works through the
    # static read-only roots, writing must stay denied by the path policy.
    workspace = tmp_path / "project"
    workspace.mkdir(exist_ok=True)
    state = ReplState(
        config=config,
        memory=memory,
        workspace=workspace,
        provider_name="local",
        provider=provider,
        model="gpt-test",
        tools_enabled=tools_enabled,
        quiet=True,
    )
    await state.new_session("t")
    return state


@pytest.fixture
async def skills_state(tmp_path: Path) -> AsyncIterator[ReplState]:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme-deploy", description="Deploy the acme stack", body="Use pdftotext.")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True)
    yield state
    await state.memory.close()


async def test_slash_status_counts_skills(skills_state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    assert await handle_slash(skills_state, "/status") is True
    assert "skills=1" in capsys.readouterr().out


async def test_skills_are_discovered_from_a_configured_root(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme-deploy")
    state = await _state(tmp_path, skills_dir=root)
    try:
        assert state.skills is not None
        assert [skill.name for skill in state.skills.skills] == ["acme-deploy"]
        assert state.instance_state.skills is state.skills
    finally:
        await state.memory.close()


async def test_slash_skills_lists_search_and_reads(
    skills_state: ReplState,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert await handle_slash(skills_state, "/skills") is True
    out = capsys.readouterr().out
    assert "skills: 1" in out
    assert "acme-deploy" in out
    assert "Deploy the acme stack" in out

    assert await handle_slash(skills_state, "/skills search pdftotext") is True
    assert "acme-deploy/SKILL.md:" in capsys.readouterr().out

    assert await handle_slash(skills_state, "/skills read acme-deploy") is True
    assert "description: Deploy the acme stack" in capsys.readouterr().out

    assert await handle_slash(skills_state, "/skills read acme-deploy --file SKILL.md") is True
    assert "acme-deploy" in capsys.readouterr().out


async def test_slash_skills_reload_picks_up_a_new_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True)
    try:
        assert await handle_slash(state, "/skills") is True
        assert "skills: 0" in capsys.readouterr().out
        _ = _write_skill(root, "added-later")
        assert await handle_slash(state, "/skills reload") is True
        assert "skills reloaded: 1 visible" in capsys.readouterr().out
        assert await handle_slash(state, "/skills") is True
        assert "added-later" in capsys.readouterr().out
    finally:
        await state.memory.close()


async def test_slash_skills_reports_a_disabled_feature(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    state = await _state(tmp_path, extra="enabled = false")
    try:
        assert await handle_slash(state, "/skills") is True
        assert "skills: disabled" in capsys.readouterr().out
        assert state.skills is None
        assert state.instance_state.skills is None
    finally:
        await state.memory.close()


async def test_slash_skills_requires_a_target(skills_state: ReplState, capsys: pytest.CaptureFixture[str]) -> None:
    for command in ("/skills search", "/skills read"):
        assert await handle_slash(skills_state, command) is True
        assert "requires" in capsys.readouterr().err


async def test_skill_tools_are_offered_and_can_read_skill_files(
    skills_state: ReplState,
) -> None:
    """The roots are read-only roots: path tools read them, writes are denied."""
    assert skills_state.agent.tools is not None
    assert skills_state.agent.tools.get("skill_list") is not None

    skill_file = skills_state.skills.skills[0].skill_file if skills_state.skills else None
    assert skill_file is not None
    read = await skills_state.agent.tools.execute("read_file", f'{{"path": "{skill_file}"}}')
    assert "Deploy the acme stack" in read

    denied = await skills_state.agent.tools.execute(
        "write_file",
        f'{{"path": "{skill_file}", "content": "x"}}',
    )
    assert denied.startswith("error:")
    assert "escapes workspace root" in denied
    assert "Deploy the acme stack" in skill_file.read_text(encoding="utf-8")


async def test_skill_tools_are_absent_when_disabled(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True, extra="enabled = false")
    try:
        assert state.agent.tools is not None
        assert state.agent.tools.get("skill_list") is None
        assert state.agent.tools.get("read_file") is not None
    finally:
        await state.memory.close()


async def test_skills_catalog_is_folded_into_the_system_prompt(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme-deploy", description="Deploy the acme stack")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True)
    try:
        prompt = state.agent.system_prompt or ""
        assert "### Skills" in prompt
        assert "- acme-deploy (config-user): Deploy the acme stack" in prompt
        assert "`skill_read`" in prompt
        # The body stays on disk: only the catalog goes into the prompt.
        assert "Body." not in prompt
    finally:
        await state.memory.close()


async def test_skills_catalog_respects_the_cap(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    for name in ("alpha", "beta", "gamma"):
        _ = _write_skill(root, name)
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True, extra="max_catalog_skills = 1")
    try:
        prompt = state.agent.system_prompt or ""
        assert "alpha" in prompt
        assert "beta" not in prompt
        assert "… and 2 more" in prompt
    finally:
        await state.memory.close()


async def test_skills_catalog_can_be_switched_off(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=True, extra="inject_catalog = false")
    try:
        assert "### Skills" not in (state.agent.system_prompt or "")
        # The tools are still offered; only the prompt block is gone.
        assert state.agent.tools is not None
        assert state.agent.tools.get("skill_list") is not None
    finally:
        await state.memory.close()


async def test_skills_catalog_needs_tools(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=False)
    try:
        assert state.agent.tools is None
        assert "### Skills" not in (state.agent.system_prompt or "")
    finally:
        await state.memory.close()


async def test_slash_skills_notes_when_tools_are_off(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    _ = _write_skill(root, "acme")
    state = await _state(tmp_path, skills_dir=root, tools_enabled=False)
    try:
        assert state.tools_enabled is False
        assert await handle_slash(state, "/skills list") is True
        assert "tools are off" in capsys.readouterr().out
    finally:
        await state.memory.close()


def test_prompt_skill_write_confirm_noninteractive_denies() -> None:
    from pathlib import Path

    from plyngent.cli.limits import prompt_skill_write_confirm

    with temporary_backend(NonInteractiveBackend()):
        decision = prompt_skill_write_confirm(Path("/skills/acme"), "create", "create skill", 1.0)
    assert decision.allow is False


def test_prompt_skill_write_confirm_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    from plyngent.cli import limits

    target = Path("/skills/acme")
    with temporary_backend(ScriptedBackend([])):
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "o\n")
        once = limits.prompt_skill_write_confirm(target, "create", "create skill", 30.0)
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "s\n")
        session = limits.prompt_skill_write_confirm(target, "create", "create skill", 30.0)
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: "n\n")
        denied = limits.prompt_skill_write_confirm(target, "create", "create skill", 30.0)
        monkeypatch.setattr(limits, "_read_yes_no_line_with_timeout", lambda _timeout: None)
        timed_out = limits.prompt_skill_write_confirm(target, "create", "create skill", 30.0)
    assert (once.allow, once.session) == (True, False)
    assert (session.allow, session.session) == (True, True)
    assert denied.allow is False
    assert timed_out.allow is False
