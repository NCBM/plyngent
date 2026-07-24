from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import active_pty_manager


def write_pty_payload(session_id: int, raw: str) -> str:
    """Write raw bytes (as str) to the PTY and format the tool status string."""
    manager = active_pty_manager()
    manager.write(session_id, raw)
    session = manager.refresh(session_id)
    exit_disp = "" if session.exit_code is None else str(session.exit_code)
    return "\n".join(
        [
            f"session_id={session_id}",
            f"alive={'true' if session.alive else 'false'}",
            f"exit_code={exit_disp}",
            f"wrote={len(raw.encode())}",
        ]
    )


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def write_pty(session_id: int, data: str) -> str:
    """Write **literal** text to a PTY session. Does not append a newline.

    ``data`` is sent unchanged (no ``ctrl+x`` / ``\\\\xHH`` expansion). For
    control sequences use :func:`write_pty_keys`.
    """
    try:
        return write_pty_payload(session_id, data)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"
