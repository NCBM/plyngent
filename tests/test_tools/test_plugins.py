"""Plugin allowlist and registration source."""

from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.catalog import ToolSource, catalog_scope, get_catalog, registration_source
from plyngent.tools.plugins import load_plugin_tools, resolve_plugin_allowlist


def test_resolve_plugin_allowlist() -> None:
    assert resolve_plugin_allowlist(None) == set()
    assert resolve_plugin_allowlist([]) == set()
    assert resolve_plugin_allowlist(["*"]) is None
    assert resolve_plugin_allowlist(["acme", " beta "]) == {"acme", "beta"}


def test_load_plugin_tools_default_loads_none() -> None:
    with catalog_scope(empty=True):
        loaded = load_plugin_tools(None)
        assert loaded == []
        assert get_catalog().names() == []


def test_plugin_registration_source_marks_entries() -> None:
    with catalog_scope(empty=True) as catalog:
        with registration_source(ToolSource(kind="plugin", plugin_id="acme")):

            @tool(name="acme_ping")
            async def acme_ping() -> str:
                return "pong"

            _ = acme_ping

        entry = catalog.get("acme_ping")
        assert entry is not None
        assert entry.source.kind == "plugin"
        assert entry.source.plugin_id == "acme"


def test_plugin_cannot_shadow_builtin_name() -> None:
    with catalog_scope(empty=True) as catalog:
        with registration_source(ToolSource(kind="builtin")):

            @tool(name="read_file")
            async def builtin_read() -> str:
                return "b"

            _ = builtin_read

        try:
            with registration_source(ToolSource(kind="plugin", plugin_id="acme")):

                @tool(name="read_file")
                async def plugin_read() -> str:
                    return "p"

                _ = plugin_read
            raise AssertionError("expected collision")
        except ValueError as exc:
            assert "collision" in str(exc)
        entry = catalog.get("read_file")
        assert entry is not None
        assert entry.source.kind == "builtin"
