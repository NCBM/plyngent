from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool
def open_pty(command: list[str], *, cwd: str = ".") -> str:
    """Open a PTY session running ``command`` (argv) under the workspace; returns session id."""
    try:
        session = PtyManager.open(command, cwd=cwd)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to open PTY: {exc}"
    return f"session_id={session.session_id}"
