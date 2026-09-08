"""MCP tool registration: namespacing, tags, catalog source, purge, handlers."""

from __future__ import annotations

import pytest

from plyngent.agent.tools import ToolTag
from plyngent.runtime.mcp_client import McpTool
from plyngent.tools.catalog import catalog_scope
from plyngent.tools.mcp import mcp_tool_name, purge_mcp_tools, register_mcp_server_tools


class _StubManager:
    """Duck-typed McpManager stand-in (records calls, returns canned text)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def call(self, server: str, tool: str, arguments: dict[str, object]) -> str:
        self.calls.append((server, tool, arguments))
        return f"{server}:{tool}:{arguments.get('text', '')}"


def _tools() -> list[McpTool]:
    return [
        McpTool(
            name="echo",
            description="Echo text.",
            input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
        ),
        McpTool(name="", description="skipped: empty name", input_schema={}),
    ]


def test_mcp_tool_name_sanitizes_and_namespaces() -> None:
    assert mcp_tool_name("my server", "a/b c") == "mcp__my_server__a_b_c"
    assert mcp_tool_name("x", "y").startswith("mcp__")


def test_register_names_tags_source_and_purge() -> None:
    manager = _StubManager()
    with catalog_scope(empty=True):
        names = register_mcp_server_tools("docs", _tools(), manager=manager)  # type: ignore[arg-type]
        assert names == ["mcp__docs__echo"]

        from plyngent.tools.catalog import get_catalog

        entry = get_catalog().get("mcp__docs__echo")
        assert entry is not None
        assert entry.source.kind == "mcp"
        assert entry.source.plugin_id == "docs"
        assert str(entry.source) == "mcp:docs"
        definition = entry.definition
        assert definition.tags == ToolTag.LOCAL
        assert definition.description.startswith("[mcp:docs]")
        assert definition.parameters["type"] == "object"
        assert "text" in definition.parameters["properties"]

        assert purge_mcp_tools() == 1
        assert get_catalog().get("mcp__docs__echo") is None
        assert purge_mcp_tools() == 0


def test_register_read_only_flag() -> None:
    manager = _StubManager()
    with catalog_scope(empty=True):
        register_mcp_server_tools("docs", _tools()[:1], manager=manager, read_only=True)  # type: ignore[arg-type]
        from plyngent.tools.catalog import get_catalog

        entry = get_catalog().get("mcp__docs__echo")
        assert entry is not None
        assert entry.definition.tags == ToolTag.LOCAL | ToolTag.READ_ONLY


def test_duplicate_registration_collides() -> None:
    manager = _StubManager()
    with catalog_scope(empty=True):
        register_mcp_server_tools("docs", _tools()[:1], manager=manager)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="collision"):
            register_mcp_server_tools("docs", _tools()[:1], manager=manager)  # type: ignore[arg-type]


async def test_handler_proxies_to_manager_call() -> None:
    manager = _StubManager()
    with catalog_scope(empty=True):
        register_mcp_server_tools("docs", _tools()[:1], manager=manager)  # type: ignore[arg-type]
        from plyngent.tools.catalog import get_catalog

        entry = get_catalog().get("mcp__docs__echo")
        assert entry is not None
        result = await entry.definition.handler(text="hi")  # type: ignore[union-attr]
        assert result == "docs:echo:hi"
        assert manager.calls == [("docs", "echo", {"text": "hi"})]
