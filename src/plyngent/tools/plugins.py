"""Load third-party tools via package entry points (allowlisted)."""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.tools.catalog import ToolSource, get_catalog, registration_source

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

ENTRY_POINT_GROUP = "plyngent.tools"


def _iter_entry_points() -> list[importlib.metadata.EntryPoint]:
    try:
        selected = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:
        # Older importlib.metadata API (unlikely on 3.14, kept for clarity).
        selected = importlib.metadata.entry_points().select(group=ENTRY_POINT_GROUP)
    return list(selected)


@dataclass(frozen=True, slots=True)
class DiscoveredPlugin:
    """One installed ``plyngent.tools`` entry point (not necessarily enabled)."""

    id: str
    value: str
    package: str | None = None
    version: str | None = None

    @property
    def module(self) -> str:
        return self.value.split(":", 1)[0] if ":" in self.value else self.value


@dataclass(frozen=True, slots=True)
class PluginStatus:
    """Discovery + allowlist status for CLI listing."""

    plugin: DiscoveredPlugin
    enabled: bool
    disabled: bool

    @property
    def will_load(self) -> bool:
        """True if current config would load this plugin."""
        return self.enabled and not self.disabled


def list_discovered_plugins() -> list[DiscoveredPlugin]:
    """Return installed entry points for :data:`ENTRY_POINT_GROUP`, sorted by id."""
    found: list[DiscoveredPlugin] = []
    for entry in _iter_entry_points():
        dist = entry.dist
        package = dist.name if dist is not None else None
        version = dist.version if dist is not None else None
        found.append(
            DiscoveredPlugin(
                id=entry.name,
                value=entry.value,
                package=package,
                version=version,
            )
        )
    return sorted(found, key=lambda p: p.id)


def plugin_would_load(
    plugin_id: str,
    *,
    enable: Sequence[str] | None,
    disable: Iterable[str] | None = None,
) -> bool:
    """Whether *plugin_id* would be loaded under the given allowlist."""
    disabled = {name.strip() for name in (disable or ()) if name.strip()}
    if plugin_id in disabled:
        return False
    allow = resolve_plugin_allowlist(enable)
    if allow is None:
        return True
    return plugin_id in allow


def list_plugin_statuses(
    *,
    enable: Sequence[str] | None,
    disable: Iterable[str] | None = None,
) -> list[PluginStatus]:
    """Discovered plugins with enable/disable/will-load flags from config lists."""
    disabled = {name.strip() for name in (disable or ()) if name.strip()}
    allow = resolve_plugin_allowlist(enable)
    rows: list[PluginStatus] = []
    for plugin in list_discovered_plugins():
        is_disabled = plugin.id in disabled
        is_enabled = True if allow is None else plugin.id in allow
        rows.append(
            PluginStatus(
                plugin=plugin,
                enabled=is_enabled and not is_disabled,
                disabled=is_disabled,
            )
        )
    return rows


def resolve_plugin_allowlist(plugins: Sequence[str] | None) -> set[str] | None:
    """Return the set of plugin ids to load, or ``None`` meaning all.

    - ``None`` / empty → load **no** plugins (default safe).
    - ``["*"]`` → load every discovered entry point.
    - otherwise → only listed entry-point names.
    """
    if not plugins:
        return set()
    if any(item.strip() == "*" for item in plugins):
        return None
    return {item.strip() for item in plugins if item.strip()}


def load_plugin_tools(
    plugins: Sequence[str] | None = None,
    *,
    disable: Iterable[str] | None = None,
) -> list[str]:
    """Import allowlisted ``plyngent.tools`` entry points under plugin sources.

    Each entry point's ``load()`` may call ``@tool``; registrations use
    ``ToolSource(kind="plugin", plugin_id=ep.name)``. Builtin name collisions
    still fail at catalog.register.

    Returns the list of plugin ids that were loaded successfully.
    """
    allow = resolve_plugin_allowlist(plugins)
    disabled = {name.strip() for name in (disable or ()) if name.strip()}
    loaded: list[str] = []
    catalog = get_catalog()
    for entry in _iter_entry_points():
        name = entry.name
        if name in disabled:
            continue
        if allow is not None and name not in allow:
            continue
        dist = entry.dist
        package = dist.name if dist is not None else None
        module = entry.value.split(":", 1)[0] if ":" in entry.value else entry.value
        source = ToolSource(
            kind="plugin",
            plugin_id=name,
            package=package,
            module=module,
        )
        with registration_source(source):
            # load() may return a callable or module; either is fine as long as
            # @tool side effects ran. Call if callable.
            obj = entry.load()
            if callable(obj):
                _ = obj()
        # Presence of any tool from this plugin is enough for "loaded".
        if any(
            entry_tool.source.kind == "plugin" and entry_tool.source.plugin_id == name
            for entry_tool in catalog.snapshot().values()
        ):
            loaded.append(name)
        else:
            # Entry point ran without registering tools — still count as loaded host-side.
            loaded.append(name)
    return loaded
