from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError

from .pty_terminal import decode_write_data
from .write_pty import write_pty_payload


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def write_pty_keys(session_id: int, data: str) -> str:
    """Write to a PTY after expanding key escapes (never for normal typing).

    Use this only for control sequences. Prefer :func:`write_pty` for plain text
    so strings like ``press ctrl+c`` are not rewritten.

    Escapes (literal characters in ``data``):

    - ``\\\\n`` ``\\\\r`` ``\\\\t`` ``\\\\e``/``\\\\E`` (ESC) ``\\\\0``
    - ``\\\\xHH`` byte, ``\\\\uHHHH`` Unicode code point
    - ``ctrl+c`` / ``ctrl+x`` ... (case-insensitive)
    - ``key=esc|enter|tab|up|down|left|right``
    """
    try:
        raw = decode_write_data(data)
        return write_pty_payload(session_id, raw)
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"
