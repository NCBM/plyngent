from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool
def write_pty(session_id: int, data: str) -> str:
    """Write text to a PTY session (interactive input). Does not append a newline."""
    try:
        PtyManager.write(session_id, data)
        session = PtyManager.refresh(session_id)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"
    exit_disp = "" if session.exit_code is None else str(session.exit_code)
    return "\n".join(
        [
            f"session_id={session_id}",
            f"alive={'true' if session.alive else 'false'}",
            f"exit_code={exit_disp}",
            f"wrote={len(data)}",
        ]
    )
