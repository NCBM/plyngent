"""Load third-party tools via package entry points (allowlisted)."""

from __future__ import annotations

import importlib.metadata
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
