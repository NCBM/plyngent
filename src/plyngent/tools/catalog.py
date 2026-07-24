"""Process-wide tool catalog: define+register vs select for model visibility."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, override

from plyngent.agent.tools import ToolDefinition, ToolTag

if TYPE_CHECKING:
    from collections.abc import Container, Generator

type ToolSourceKind = Literal["builtin", "plugin"]
type ToolSurface = Literal["local", "public"]


@dataclass(frozen=True, slots=True)
class ToolSource:
    """Where a catalog entry came from (not a :class:`~plyngent.agent.tools.ToolTag`)."""

    kind: ToolSourceKind
    plugin_id: str | None = None
    package: str | None = None
    module: str | None = None

    @override
    def __str__(self) -> str:
        if self.kind == "builtin":
            return "builtin"
        plugin = self.plugin_id or "?"
        return f"plugin:{plugin}"


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    definition: ToolDefinition
    source: ToolSource


_BUILTIN_SOURCE = ToolSource(kind="builtin")
_registration_source: ContextVar[ToolSource] = ContextVar(
    "plyngent_tool_registration_source",
    default=_BUILTIN_SOURCE,
)


def get_registration_source() -> ToolSource:
    return _registration_source.get()


@contextmanager
def registration_source(source: ToolSource) -> Generator[None]:
    """Set :func:`get_registration_source` for the duration of a load block."""
    token = _registration_source.set(source)
    try:
        yield
    finally:
        _registration_source.reset(token)


class ToolCatalog:
    """Name → registered tool map for one process (or test-scoped snapshot)."""

    _by_name: dict[str, RegisteredTool]

    def __init__(self) -> None:
        self._by_name = {}

    def register(self, definition: ToolDefinition, *, source: ToolSource | None = None) -> None:
        """Add *definition*; never shadow an existing name (builtin or plugin)."""
        resolved = source if source is not None else get_registration_source()
        existing = self._by_name.get(definition.name)
        if existing is not None:
            msg = (
                f"tool name collision: {definition.name!r} already registered "
                f"from {existing.source} (refusing {resolved})"
            )
            raise ValueError(msg)
        self._by_name[definition.name] = RegisteredTool(definition=definition, source=resolved)

    def get(self, name: str) -> RegisteredTool | None:
        return self._by_name.get(name)

    def names(self) -> list[str]:
        return sorted(self._by_name)

    @staticmethod
    def _matches(  # noqa: PLR0911 — filter predicates are clearer as early exits
        entry: RegisteredTool,
        *,
        surface: ToolSurface,
        sources: Container[ToolSourceKind] | None,
        plugin_ids: Container[str] | None,
        require_tags: ToolTag | None,
        exclude_tags: ToolTag | None,
        include_names: set[str] | None,
        exclude_names: set[str] | None,
    ) -> bool:
        name = entry.definition.name
        if include_names is not None and name not in include_names:
            return False
        if exclude_names is not None and name in exclude_names:
            return False
        if sources is not None and entry.source.kind not in sources:
            return False
        if plugin_ids is not None:
            plugin_id = entry.source.plugin_id
            if entry.source.kind != "plugin" or plugin_id is None or plugin_id not in plugin_ids:
                return False
        tags = entry.definition.tags
        if surface == "public" and not (tags & ToolTag.PUBLIC):
            return False
        if surface == "local" and not (tags & (ToolTag.LOCAL | ToolTag.PUBLIC)):
            return False
        if require_tags is not None and (tags & require_tags) != require_tags:
            return False
        return not (exclude_tags is not None and tags & exclude_tags)

    def select(
        self,
        *,
        surface: ToolSurface = "local",
        sources: Container[ToolSourceKind] | None = None,
        plugin_ids: Container[str] | None = None,
        require_tags: ToolTag | None = None,
        exclude_tags: ToolTag | None = None,
        include_names: set[str] | None = None,
        exclude_names: set[str] | None = None,
    ) -> list[ToolDefinition]:
        """Return definitions allowed for this host surface / filters.

        *surface* ``local`` keeps tools with LOCAL and/or PUBLIC.
        *surface* ``public`` keeps only tools with PUBLIC.
        """
        return [
            self._by_name[name].definition
            for name in sorted(self._by_name)
            if self._matches(
                self._by_name[name],
                surface=surface,
                sources=sources,
                plugin_ids=plugin_ids,
                require_tags=require_tags,
                exclude_tags=exclude_tags,
                include_names=include_names,
                exclude_names=exclude_names,
            )
        ]

    def clear(self) -> None:
        self._by_name.clear()

    def snapshot(self) -> dict[str, RegisteredTool]:
        return dict(self._by_name)

    def restore(self, snapshot: dict[str, RegisteredTool]) -> None:
        self._by_name = dict(snapshot)


_catalog = ToolCatalog()
_catalog_override: ContextVar[ToolCatalog | None] = ContextVar(
    "plyngent_tool_catalog_override",
    default=None,
)


def get_catalog() -> ToolCatalog:
    override = _catalog_override.get()
    if override is not None:
        return override
    return _catalog


def register_tool(definition: ToolDefinition, *, source: ToolSource | None = None) -> None:
    get_catalog().register(definition, source=source)


@contextmanager
def catalog_scope(*, empty: bool = True) -> Generator[ToolCatalog]:
    """Snapshot the process catalog; optionally start empty for unit tests.

    Restores the previous catalog contents on exit. Nested scopes stack via
    contextvars when *empty* installs a fresh catalog override.
    """
    if empty:
        scoped = ToolCatalog()
        token: Token[ToolCatalog | None] = _catalog_override.set(scoped)
        try:
            yield scoped
        finally:
            _catalog_override.reset(token)
        return

    catalog = get_catalog()
    snap = catalog.snapshot()
    try:
        yield catalog
    finally:
        catalog.restore(snap)


_builtins_loaded = False


def _ensure_builtin_definitions_registered(catalog: ToolCatalog) -> None:
    """Register builtin ToolDefinitions if the active catalog is missing them.

    Importing tool modules only registers into the catalog that was active at
    import time (usually the process catalog). Test ``catalog_scope(empty=True)``
    overrides must still see builtins for ``default_tool_definitions``.
    """
    from plyngent.tools.chat import CHAT_TOOLS
    from plyngent.tools.file import FILE_TOOLS
    from plyngent.tools.process import PROCESS_TOOLS
    from plyngent.tools.todo import TODO_TOOLS
    from plyngent.tools.vcs import VCS_TOOLS

    for definition in (*FILE_TOOLS, *PROCESS_TOOLS, *VCS_TOOLS, *CHAT_TOOLS, *TODO_TOOLS):
        if catalog.get(definition.name) is None:
            catalog.register(definition, source=_BUILTIN_SOURCE)


def register_builtin_tools(*, force: bool = False) -> ToolCatalog:
    """Import builtin tool modules so ``@tool`` registrations run.

    Idempotent unless *force* is true (re-import path still no-ops if names
    already exist — callers should use :func:`catalog_scope` for isolation).
    Always ensures the **active** catalog (including overrides) has builtins.
    """
    global _builtins_loaded  # noqa: PLW0603 — process load flag
    if not _builtins_loaded or force:
        # Import side effects: each module's @tool(...) registers into the catalog.
        # Group packages pull their leaf modules (FILE_TOOLS etc. still exported).
        import plyngent.tools.chat as _chat_tools
        import plyngent.tools.file as _file_tools
        import plyngent.tools.process as _process_tools
        import plyngent.tools.temp_workspace as _temp_workspace_tools
        import plyngent.tools.todo as _todo_tools
        import plyngent.tools.vcs as _vcs_tools

        _ = (
            _chat_tools,
            _file_tools,
            _process_tools,
            _temp_workspace_tools,
            _todo_tools,
            _vcs_tools,
        )
        _builtins_loaded = True

    catalog = get_catalog()
    _ensure_builtin_definitions_registered(catalog)
    return catalog


def default_tool_definitions(
    *,
    surface: ToolSurface = "local",
    sources: Container[ToolSourceKind] | None = None,
) -> list[ToolDefinition]:
    """Load builtins (if needed) and select for the given surface."""
    catalog = register_builtin_tools()
    kind_filter: Container[ToolSourceKind] | None = sources
    if kind_filter is None:
        kind_filter = ("builtin",)
    return catalog.select(surface=surface, sources=kind_filter)
