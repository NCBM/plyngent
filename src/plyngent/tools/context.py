"""Instance / session tool context (contextvars) for state affinity tags."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from plyngent.tools.view import PersistentDataView, session_data_view

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from plyngent.agent.todo_stack import TodoStack
    from plyngent.tools.workspace import WorkspacePolicy


def _default_instance_data() -> PersistentDataView[Any]:
    return session_data_view()


def _default_session_data() -> PersistentDataView[Any]:
    return session_data_view()


def _default_workspace_policy() -> WorkspacePolicy:
    from plyngent.tools.workspace import WorkspacePolicy

    return WorkspacePolicy()


@dataclass
class InstanceState:
    """Process / agent-host scoped state for INSTANCE_STATE tools."""

    # Primary workspace root facet (mirrors workspace.policy.root when set).
    workspace_root: Path | None = None
    # Path/command policy bag for this host (preferred over process globals).
    workspace: WorkspacePolicy = field(default_factory=_default_workspace_policy)
    data: PersistentDataView[Any] = field(default_factory=_default_instance_data)
    # Ephemeral process maps (PTY etc.) may hang here later.
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.workspace_root is not None and self.workspace.root is None:
            self.workspace.root = self.workspace_root
        elif self.workspace.root is not None and self.workspace_root is None:
            self.workspace_root = self.workspace.root

    @property
    def pty(self) -> Any:
        """PTY manager facade for this process (class-level sessions today)."""
        from plyngent.tools.process.pty_session import PtyManager

        return PtyManager

    async def shutdown(self) -> None:
        """Best-effort cleanup hooks (PTY close, temp workspaces)."""
        from plyngent.tools.temp_workspace import cleanup_temporary_workspaces

        self.pty.close_all()
        _ = cleanup_temporary_workspaces()


@dataclass
class SessionState:
    """One chat session for SESSION_STATE tools."""

    session_id: int | str | None = None
    data: PersistentDataView[Any] = field(default_factory=_default_session_data)
    # Live domain object (also under data["todo"] when views bind).
    todo: TodoStack | None = None
    # Host hook after todo tools mutate (e.g. CLI memory persist).
    on_todo_change: Callable[[], None] | None = None
    # Soft-confirm grants live map: tool_name → True (Phase 1 key is tool name).
    # Durable copy lives under data["grants"] (see plyngent.tools.grants).
    grants: dict[str, bool] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def has_grant(self, tool_name: str) -> bool:
        return bool(self.grants.get(tool_name))

    def add_grant(self, tool_name: str) -> None:
        """Update the live map only; prefer :func:`plyngent.tools.grants.add_grant` to persist."""
        self.grants[tool_name] = True

    def clear_grants(self) -> None:
        """Clear the live map; durable view is updated separately when needed."""
        self.grants.clear()


_instance: ContextVar[InstanceState | None] = ContextVar("plyngent_instance_state", default=None)
_session: ContextVar[SessionState | None] = ContextVar("plyngent_session_state", default=None)


def get_instance() -> InstanceState | None:
    return _instance.get()


def get_session() -> SessionState | None:
    return _session.get()


def require_instance() -> InstanceState:
    state = _instance.get()
    if state is None:
        msg = "instance state is not bound; host must set InstanceState around tool execution"
        raise RuntimeError(msg)
    return state


def require_session() -> SessionState:
    state = _session.get()
    if state is None:
        msg = "session state is not bound; host must set SessionState around tool execution"
        raise RuntimeError(msg)
    return state


@contextmanager
def bind_instance(state: InstanceState | None) -> Generator[InstanceState | None]:
    token: Token[InstanceState | None] = _instance.set(state)
    try:
        yield state
    finally:
        _instance.reset(token)


@contextmanager
def bind_session(state: SessionState | None) -> Generator[SessionState | None]:
    token: Token[SessionState | None] = _session.set(state)
    try:
        yield state
    finally:
        _session.reset(token)


@contextmanager
def bind_tool_context(
    *,
    instance: InstanceState | None = None,
    session: SessionState | None = None,
) -> Generator[tuple[InstanceState | None, SessionState | None]]:
    """Bind instance and session contextvars for a tool batch / test."""
    with bind_instance(instance), bind_session(session):
        yield instance, session
