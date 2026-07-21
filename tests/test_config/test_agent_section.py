from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOOL_DIRECTIVES,
    compose_agent_system_content,
    load,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_agent_section_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    store = load(path)
    assert store.agent_config.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert store.agent_config.tool_directives == DEFAULT_TOOL_DIRECTIVES
    assert "professional coding agent" in store.agent_config.system_prompt
    assert "### Workspace" in store.agent_config.tool_directives
    assert store.agent_config.max_tool_result_chars == 32_000
    assert store.agent_config.parallel_tools is True
    assert store.agent_config.confirm_destructive is True
    assert store.agent_config.path_denylist == []
    assert store.agent_config.max_context_tokens == 200_000


def test_compose_defaults_join_persona_and_directives() -> None:
    body = compose_agent_system_content(DEFAULT_SYSTEM_PROMPT, DEFAULT_TOOL_DIRECTIVES)
    assert body is not None
    assert body.startswith("You are a professional coding agent")
    assert "\n\n### Workspace" in body
    assert "### Todos" in body
    # No double-blank-line collapse issues between parts.
    assert "\n\n\n" not in body


def test_compose_empty_combinations() -> None:
    assert compose_agent_system_content("", "") is None
    assert compose_agent_system_content("  ", "\n") is None
    assert compose_agent_system_content("Only persona", "") == "Only persona"
    assert compose_agent_system_content("", "Only tools") == "Only tools"
    assert compose_agent_system_content("Persona", "Tools") == "Persona\n\nTools"


def test_agent_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
system_prompt = "Be brief."
tool_directives = "Use tools carefully."
max_tool_result_chars = 100
parallel_tools = false
confirm_destructive = false
path_denylist = ["/secrets/", ".ssh/"]
max_context_tokens = 5000
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.agent_config.system_prompt == "Be brief."
    assert store.agent_config.tool_directives == "Use tools carefully."
    assert store.agent_config.max_tool_result_chars == 100
    assert store.agent_config.parallel_tools is False
    assert store.agent_config.confirm_destructive is False
    assert store.agent_config.path_denylist == ["/secrets/", ".ssh/"]
    assert store.agent_config.max_context_tokens == 5000
    composed = compose_agent_system_content(
        store.agent_config.system_prompt,
        store.agent_config.tool_directives,
    )
    assert composed == "Be brief.\n\nUse tools carefully."


def test_agent_system_prompt_empty_disables_persona_only(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
system_prompt = ""
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.agent_config.system_prompt == ""
    assert store.agent_config.tool_directives == DEFAULT_TOOL_DIRECTIVES
    body = compose_agent_system_content(
        store.agent_config.system_prompt,
        store.agent_config.tool_directives,
    )
    assert body is not None
    assert "### Workspace" in body
    assert "professional coding agent" not in body


def test_agent_tool_directives_empty_disables_playbook_only(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
tool_directives = ""
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.agent_config.tool_directives == ""
    assert store.agent_config.system_prompt == DEFAULT_SYSTEM_PROMPT
    body = compose_agent_system_content(
        store.agent_config.system_prompt,
        store.agent_config.tool_directives,
    )
    assert body is not None
    assert "professional coding agent" in body
    assert "### Workspace" not in body


def test_agent_both_empty_disables_system(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
system_prompt = ""
tool_directives = ""
""",
        encoding="utf-8",
    )
    store = load(path)
    assert (
        compose_agent_system_content(
            store.agent_config.system_prompt,
            store.agent_config.tool_directives,
        )
        is None
    )
