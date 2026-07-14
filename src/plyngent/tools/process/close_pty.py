from __future__ import annotations

from plyngent.agent import tool

from .pty_session import PtyManager


@tool
def close_pty(session_id: int) -> str:
    """Close a PTY session and terminate its process."""
    PtyManager.close(session_id)
    return f"closed session_id={session_id}"
