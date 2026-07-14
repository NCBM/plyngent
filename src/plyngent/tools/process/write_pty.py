from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool
def write_pty(session_id: int, data: str) -> str:
    """Write text to a PTY session (e.g. interactive input). Does not append a newline."""
    try:
        PtyManager.write(session_id, data)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"
    return f"wrote {len(data)} characters to session_id={session_id}"
