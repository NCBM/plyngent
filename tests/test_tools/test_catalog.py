"""Tool catalog, tags, and default select parity."""

from __future__ import annotations

import inspect

from plyngent.agent import ToolTag, tool
from plyngent.tools import DEFAULT_TOOLS, default_tool_definitions, register_builtin_tools
from plyngent.tools.catalog import ToolCatalog, ToolSource, catalog_scope, get_catalog, registration_source


def test_default_tool_names_match_legacy_lists() -> None:
    register_builtin_tools()
    selected = default_tool_definitions(surface="local")
    legacy = [t.name for t in DEFAULT_TOOLS]
    assert sorted(t.name for t in selected) == sorted(legacy)
    assert len(selected) == len(legacy)


def test_all_builtins_are_async_and_tagged() -> None:
    tools = default_tool_definitions()
    assert tools
    for definition in tools:
        assert inspect.iscoroutinefunction(definition.handler), definition.name
        assert definition.tags & (ToolTag.LOCAL | ToolTag.PUBLIC)
        assert definition.tags & ToolTag.LOCAL or definition.tags & ToolTag.PUBLIC


def test_public_surface_is_subset() -> None:
    local = {t.name for t in default_tool_definitions(surface="local")}
    public = {t.name for t in default_tool_definitions(surface="public")}
    assert public
    assert public <= local
    # Todo series is the main PUBLIC surface today.
    assert "todo_list" in public
    assert "read_file" not in public


def test_catalog_scope_empty_isolates_registration() -> None:
    with catalog_scope(empty=True) as catalog:

        @tool(name="only_in_scope", register=True)
        async def only_in_scope() -> str:
            return "ok"

        _ = only_in_scope
        assert catalog.get("only_in_scope") is not None
        # default_tool_definitions re-seeds builtins into the override; plugin-like
        # names from this scope stay local to the override and leave process catalog.
    assert get_catalog().get("only_in_scope") is None


def test_collision_refuses_shadow() -> None:
    catalog = ToolCatalog()

    @tool(register=False)
    async def alpha() -> str:
        return "a"

    catalog.register(alpha, source=ToolSource(kind="builtin"))
    try:
        catalog.register(alpha, source=ToolSource(kind="plugin", plugin_id="acme"))
        raise AssertionError("expected collision")
    except ValueError as exc:
        assert "collision" in str(exc)


def test_registration_source_context() -> None:
    with catalog_scope(empty=True) as catalog:
        with registration_source(ToolSource(kind="plugin", plugin_id="acme")):

            @tool(name="plugin_tool")
            async def plugin_tool() -> str:
                return "p"

            _ = plugin_tool

        entry = catalog.get("plugin_tool")
        assert entry is not None
        assert entry.source.kind == "plugin"
        assert entry.source.plugin_id == "acme"


def test_tool_tags_reject_neither_surface() -> None:
    try:

        @tool(tags=ToolTag.YOLO, register=False)  # type: ignore[arg-type]
        async def bad() -> str:
            return "x"

        _ = bad
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "LOCAL" in str(exc)
