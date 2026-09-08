"""Register MCP server tools into the tool catalog (namespaced, LOCAL tags).

Mirrors :mod:`plyngent.tools.plugins`: the catalog only *registers* what the
model may see; hosts still select into a :class:`ToolRegistry`. Connections
must already be started (see :class:`~plyngent.runtime.mcp_client.McpManager`);
this module is sync-only so registry rebuilds never need an event loop.

Tool names are namespaced ``mcp__<server>__<tool>`` so two servers (or a
server and a builtin) can never collide on sanitized names; the catalog still
refuses true clashes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.agent.tools import ToolDefinition, ToolTag
from plyngent.tools.catalog import ToolSource, get_catalog

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.config.models import McpConfig
    from plyngent.runtime.mcp_client import McpManager, McpTool

_MCP_PREFIX = "mcp__"
_NAME_SEGMENT_MAX = 64
_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9_-]")


def _sanitize_segment(part: str) -> str:
    """Reduce a server/tool name to a model-safe, bounded name segment."""
    return _SANITIZE_PATTERN.sub("_", part)[:_NAME_SEGMENT_MAX] or "srv"


def mcp_tool_name(server: str, tool: str) -> str:
    """Namespaced catalog name for one MCP tool (collision-free per server)."""
    return f"{_MCP_PREFIX}{_sanitize_segment(server)}__{_sanitize_segment(tool)}"


def _make_handler(
    manager: McpManager,
    server: str,
    tool: str,
) -> Callable[..., Awaitable[str]]:
    async def handler(**arguments: object) -> str:
        return await manager.call(server, tool, dict(arguments))

    return handler


def register_mcp_server_tools(
    server: str,
    tools: list[McpTool],
    *,
    manager: McpManager,
    read_only: bool = False,
) -> list[str]:
    """Register one server's advertised tools; returns the registered names.

    Tags: LOCAL (plus READ_ONLY when the server config opts in) — never YOLO or
    TRUSTABLE, so host-side danger classification always sees every call.
    """
    catalog = get_catalog()
    source = ToolSource(kind="mcp", plugin_id=server)
    names: list[str] = []
    for entry in tools:
        if not entry.name:
            continue
        tags = ToolTag.LOCAL
        if read_only:
            tags |= ToolTag.READ_ONLY
        description = entry.description or f"MCP tool {entry.name!r} from server {server!r}"
        definition = ToolDefinition(
            name=mcp_tool_name(server, entry.name),
            description=f"[mcp:{server}] {description}",
            parameters=dict(entry.input_schema),
            handler=_make_handler(manager, server, entry.name),
            tags=tags,
        )
        catalog.register(definition, source=source)
        names.append(definition.name)
    return names


def purge_mcp_tools() -> int:
    """Remove previously registered MCP tools (call before re-registering)."""
    return get_catalog().purge_kind("mcp")


def register_mcp_tools(manager: McpManager, config: McpConfig) -> list[str]:
    """Register every connected server's tools; replaces prior MCP entries."""
    _ = purge_mcp_tools()
    names: list[str] = []
    for connection in manager.connections():
        if not connection.connected or not connection.tools:
            continue
        server_cfg = config.servers.get(connection.name)
        names.extend(
            register_mcp_server_tools(
                connection.name,
                connection.tools,
                manager=manager,
                read_only=server_cfg.read_only if server_cfg is not None else False,
            )
        )
    return names
