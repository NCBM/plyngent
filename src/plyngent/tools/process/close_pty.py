from __future__ import annotations

from plyngent.agent import tool

from .pty_session import PtyManager, format_close_result


@tool
def close_pty(session_id: int) -> str:
    """Close a PTY session (SIGTERM, then SIGKILL after a short grace period)."""
    result = PtyManager.close(session_id)
    return format_close_result(result)
