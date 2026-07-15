from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.config import load

if TYPE_CHECKING:
    from pathlib import Path


def test_agent_section_defaults(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text("", encoding="utf-8")
    store = load(path)
    assert store.agent_config.system_prompt == ""
    assert store.agent_config.max_tool_result_chars == 32_000
    assert store.agent_config.parallel_tools is True
    assert store.agent_config.confirm_destructive is True
    assert store.agent_config.path_denylist == []
    assert store.agent_config.max_context_chars == 200_000


def test_agent_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
system_prompt = "Be brief."
max_tool_result_chars = 100
parallel_tools = false
confirm_destructive = false
path_denylist = ["/secrets/", ".ssh/"]
max_context_chars = 5000
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.agent_config.system_prompt == "Be brief."
    assert store.agent_config.max_tool_result_chars == 100
    assert store.agent_config.parallel_tools is False
    assert store.agent_config.confirm_destructive is False
    assert store.agent_config.path_denylist == ["/secrets/", ".ssh/"]
    assert store.agent_config.max_context_chars == 5000
