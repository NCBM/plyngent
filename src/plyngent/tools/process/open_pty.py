from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, argv_shape_error

from .pty_session import active_pty_manager

if TYPE_CHECKING:
    from collections.abc import Mapping


def open_pty_shape(args: Mapping[str, object]) -> str | None:
    """Answer a session that cannot start, before anyone is asked about it."""
    shape_error = argv_shape_error(args.get("command"))
    if shape_error is not None:
        return f"error: `open_pty` {shape_error}"
    return None


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, validate=open_pty_shape)
async def open_pty(command: list[str], *, cwd: str = ".") -> str:
    """Open a PTY session running ``command`` (argv) under the workspace.

    POSIX uses openpty/fork; Windows uses ConPTY (pywinpty). Returns structured
    status including session_id. Failed exec may surface via later read_pty
    data (marker) and a non-zero exit_code.
    """
    try:
        session = active_pty_manager().open(command, cwd=cwd)
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
