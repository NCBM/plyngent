"""Host-terminal safety helpers for agent PTY sessions.

Child PTY I/O is isolated on the master FD. The practical leak is that
``read_pty`` returns raw CSI/alt-screen sequences which the CLI then echoes
to the *user's* stdout as tool results — those bytes reprogram the host TTY.
"""

from __future__ import annotations

import contextlib
import re
import sys
from typing import Final

# Leave alt-screen, restore cursor visibility, reset SGR, keypad, mouse, etc.
# Intentional over-reset: cheap and safe after TUI children.
_HOST_RESTORE: Final[str] = (
    "\x1b[?1049l"  # leave alternate screen buffer
    "\x1b[?47l"  # leave alt screen (legacy)
    "\x1b[?25h"  # show cursor
    "\x1b[0m"  # reset SGR
    "\x1b[?1l"  # normal cursor keys
    "\x1b[?1000l"  # mouse off
    "\x1b[?1002l"
    "\x1b[?1003l"
    "\x1b[?1006l"
    "\x1b[?2004l"  # bracketed paste off
    "\x1b[?7h"  # wraparound on
    "\r\n"
)

_HEX_BYTE = re.compile(r"\\x([0-9A-Fa-f]{2})")
_HEX_U = re.compile(r"\\u([0-9A-Fa-f]{4})")
_CTRL = re.compile(r"ctrl\+([a-z@\[\\\]\^_])", re.IGNORECASE)
_KEY = re.compile(r"key=(esc|escape|enter|tab|up|down|left|right)", re.IGNORECASE)
_SIMPLE = {
    r"\n": "\n",
    r"\r": "\r",
    r"\t": "\t",
    r"\e": "\x1b",
    r"\E": "\x1b",
    r"\0": "\x00",
}

_KEY_MAP = {
    "esc": "\x1b",
    "escape": "\x1b",
    "enter": "\r",
    "tab": "\t",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
}
_SIMPLE_ESCAPE_LEN = 2


def restore_host_terminal() -> None:
    """Best-effort reset of the *user* terminal (stdout), if it is a TTY."""
    try:
        out = sys.stdout
        if not out.isatty():
            return
        buffer = getattr(out, "buffer", None)
        if buffer is not None:
            with contextlib.suppress(OSError, ValueError):
                _ = buffer.write(_HOST_RESTORE.encode("ascii", errors="replace"))
                _ = buffer.flush()
                return
        with contextlib.suppress(OSError, ValueError):
            _ = out.write(_HOST_RESTORE)
            _ = out.flush()
    except AttributeError, OSError, ValueError:
        return


def sanitize_pty_output_for_tool(data: str) -> str:
    """Make PTY bytes safe to print on the host as tool/chat text.

    Escapes ESC (``\\\\x1b``) so CSI sequences are visible text instead of
    control codes when the CLI echoes tool results.
    """
    if not data or "\x1b" not in data:
        return data
    return data.replace("\x1b", "\\x1b")


def decode_write_data(data: str) -> str:
    """Expand agent-friendly escapes into raw PTY input.

    Supported (as literal characters in ``data``):

    - ``\\n`` ``\\r`` ``\\t`` ``\\e`` / ``\\E`` (ESC) ``\\0``
    - ``\\xHH`` byte, ``\\uHHHH`` Unicode code point
    - ``ctrl+c`` / ``ctrl+x`` ... (case-insensitive; A-Z or @ [ \\ ] ^ _)
    - ``key=esc`` ``key=enter`` ``key=tab`` ``key=up|down|left|right``
    """
    if not data:
        return data

    # Order: multi-char named tokens first, then hex, then simple backslash pairs.
    out: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        rest = data[i:]
        m_key = _KEY.match(rest)
        if m_key is not None:
            out.append(_KEY_MAP[m_key.group(1).lower()])
            i += m_key.end()
            continue
        m_ctrl = _CTRL.match(rest)
        if m_ctrl is not None:
            ch = m_ctrl.group(1).upper()
            out.append(chr((ord(ch) - ord("@")) & 0x1F))
            i += m_ctrl.end()
            continue
        m_hex = _HEX_BYTE.match(rest)
        if m_hex is not None:
            out.append(chr(int(m_hex.group(1), 16)))
            i += m_hex.end()
            continue
        m_u = _HEX_U.match(rest)
        if m_u is not None:
            out.append(chr(int(m_u.group(1), 16)))
            i += m_u.end()
            continue
        if rest.startswith("\\") and len(rest) >= _SIMPLE_ESCAPE_LEN:
            pair = rest[:_SIMPLE_ESCAPE_LEN]
            if pair in _SIMPLE:
                out.append(_SIMPLE[pair])
                i += _SIMPLE_ESCAPE_LEN
                continue
        out.append(data[i])
        i += 1
    return "".join(out)
