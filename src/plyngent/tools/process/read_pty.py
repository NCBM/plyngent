from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool
def read_pty(session_id: int, *, max_bytes: int = 8192, timeout: float = 0.2) -> str:
    """Read available output from a PTY session (may be empty if nothing ready)."""
    try:
        return PtyManager.read(session_id, max_bytes=max_bytes, timeout=timeout)
    except WorkspaceError as exc:
        return f"error: {exc}"
