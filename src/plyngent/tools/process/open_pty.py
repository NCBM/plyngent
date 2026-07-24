from __future__ import annotations

import shlex

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError

from .pty_session import PtyManager


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def open_pty(command: list[str], *, cwd: str = ".") -> str:
    """Open a PTY session running ``command`` (argv) under the workspace.

    POSIX uses openpty/fork; Windows uses ConPTY (pywinpty). Returns structured
    status including session_id. Failed exec may surface via later read_pty
    data (marker) and a non-zero exit_code.
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
            f"cmd={shlex.join(session.command)}",
        ]
    )
