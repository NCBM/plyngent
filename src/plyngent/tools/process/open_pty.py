from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool
def open_pty(command: list[str], *, cwd: str = ".") -> str:
    """Open a Unix PTY session running ``command`` (argv) under the workspace.

    Returns structured status including session_id. Not supported on Windows.
    Failed exec surfaces via later read_pty data (marker) and exit_code=127.
    """
    try:
        session = PtyManager.open(command, cwd=cwd)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to open PTY: {exc}"
    return "\n".join(
        [
            f"session_id={session.session_id}",
            "alive=true",
            "exit_code=",
            f"cmd={' '.join(session.command)}",
        ]
    )
