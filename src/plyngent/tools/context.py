"""Instance / session tool context (contextvars) for state affinity tags."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from plyngent.tools.view import PersistentDataView, session_data_view

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from plyngent.agent.todo_stack import TodoStack


def _default_instance_data() -> PersistentDataView[Any]:
    return session_data_view()


def _default_session_data() -> PersistentDataView[Any]:
    return session_data_view()


@dataclass
class InstanceState:
    """Process / agent-host scoped state for INSTANCE_STATE tools."""

    # Optional fixed facets (workspace path still also lives in workspace module
    # during migration; hosts should set both until globals retire).
    workspace_root: Path | None = None
    data: PersistentDataView[Any] = field(default_factory=_default_instance_data)
    # Ephemeral process maps (PTY etc.) may hang here later.
    extras: dict[str, Any] = field(default_factory=dict)

    async def shutdown(self) -> None:
        """Best-effort cleanup hooks (PTY close, temp workspaces)."""
        from plyngent.tools.process.pty_session import PtyManager
        from plyngent.tools.temp_workspace import cleanup_temporary_workspaces

        PtyManager.close_all()
        _ = cleanup_temporary_workspaces()


@dataclass
class SessionState:
    """One chat session for SESSION_STATE tools."""

    session_id: int | str | None = None
    data: PersistentDataView[Any] = field(default_factory=_default_session_data)
    # Live domain object during migration (also under data["todo"] when views bind).
    todo: TodoStack | None = None
    # Soft-confirm grants: tool_name → True (Phase 1 key is tool name in session).
    grants: dict[str, bool] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def has_grant(self, tool_name: str) -> bool:
        return bool(self.grants.get(tool_name))

    def add_grant(self, tool_name: str) -> None:
        self.grants[tool_name] = True

    def clear_grants(self) -> None:
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
