from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.shape import shape_of
from plyngent.tools.workspace import WorkspaceError

from .pty_terminal import KeySyntaxError, decode_write_data
from .write_pty import write_pty_payload

if TYPE_CHECKING:
    from collections.abc import Mapping


def write_pty_keys_shape(args: Mapping[str, object]) -> str | None:
    """Answer a bad marker list, before anyone is asked about the call."""
    data = args.get("data")
    if not isinstance(data, str):
        return f"error: `write_pty_keys` expects `data` as a string, got {shape_of(data)}"
    try:
        _ = decode_write_data(data)
    except KeySyntaxError as exc:
        return f"error: {exc}"
    return None


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, validate=write_pty_keys_shape)
async def write_pty_keys(session_id: int, data: str) -> str:
    """Write to a PTY after expanding ``<key>`` markers (never for normal typing).

    Use this only for control sequences. Prefer :func:`write_pty` for plain text
    so strings like ``press <c-c>`` are not rewritten.

    Markers, written between angle brackets:

    - ``<esc>`` ``<enter>`` ``<tab>`` ``<s-tab>`` ``<space>`` ``<bs>`` ``<del>``
      ``<ins>`` ``<home>`` ``<end>`` ``<up>`` ``<down>`` ``<left>`` ``<right>``
      ``<pgup>`` ``<pgdn>`` ``<f1>`` … ``<f12>``
    - a modifier in front, alone or combined: ``<c-up>`` ``<s-home>``
      ``<m-tab>`` ``<c-s-f5>``; on a plain character Ctrl is its control byte
      (``<c-a>``) and Alt an ESC before it (``<m-x>``)
    - ``<up*3>`` repeats a key, ``<<`` is a literal ``<``

    An unknown marker is an error, not text typed into the terminal: a typo must
    not land in a shell prompt.

    Escapes (as the characters ``data`` carries, so doubled inside JSON):
    ``\\\\n`` ``\\\\r`` ``\\\\t`` ``\\\\e``/``\\\\E`` (ESC) ``\\\\0``,
    ``\\\\xHH`` byte, ``\\\\uHHHH`` code point. Anything else is literal.
    """
    try:
        raw = decode_write_data(data)
        return write_pty_payload(session_id, raw)
    except KeySyntaxError as exc:
        return f"error: {exc}"
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: failed to write PTY: {exc}"
