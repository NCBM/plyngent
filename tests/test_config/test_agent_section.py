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
    assert store.agent_config.max_tool_result_chars == 32_000  # noqa: PLR2004
    assert store.agent_config.parallel_tools is True


def test_agent_section_parse(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    _ = path.write_text(
        """
[agent]
system_prompt = "Be brief."
max_tool_result_chars = 100
parallel_tools = false
""",
        encoding="utf-8",
    )
    store = load(path)
    assert store.agent_config.system_prompt == "Be brief."
    assert store.agent_config.max_tool_result_chars == 100  # noqa: PLR2004
    assert store.agent_config.parallel_tools is False
