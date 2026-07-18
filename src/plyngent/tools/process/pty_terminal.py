"""PTY payload helpers for agent tools.

Child PTY I/O stays on the master FD. Tool results are printed on the *user*
TTY by the CLI, so any CSI/control bytes in ``read_pty`` data would reprogram
the host terminal. We sanitize on the tool boundary instead of resetting the
host after close (which flashes the screen on every chat exit).
"""

from __future__ import annotations

import re

# Match common \xNN / \uNNNN / named ctrl+ / esc sequences in write_pty_keys data.
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

# C0 controls except TAB/LF/CR (those stay for readable logs).
_UNSAFE_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_pty_output_for_tool(data: str) -> str:
    """Make PTY bytes safe to print on the host as tool/chat text.

    - ESC becomes the two-character sequence ``\\\\x1b`` so CSI is not executed
      when the CLI echoes tool results.
    - Other C0 controls (except tab/LF/CR) become ``\\\\xHH``.
    """
    if not data:
        return data

    def _esc_ctrl(match: re.Match[str]) -> str:
        return f"\\x{ord(match.group(0)):02x}"

    # ESC first as a stable, readable form used in docs/tests.
    out = data.replace("\x1b", "\\x1b")
    return _UNSAFE_CTRL.sub(_esc_ctrl, out)


def decode_write_data(data: str) -> str:
    """Expand agent-friendly escapes into raw PTY input (for ``write_pty_keys`` only).

    Supported (as literal characters in ``data``):

    - ``\\n`` ``\\r`` ``\\t`` ``\\e`` / ``\\E`` (ESC) ``\\0``
    - ``\\xHH`` byte, ``\\uHHHH`` Unicode code point
    - ``ctrl+c`` / ``ctrl+x`` ... (case-insensitive; A-Z or @ [ \\ ] ^ _)
    - ``key=esc`` ``key=enter`` ``key=tab`` ``key=up|down|left|right``
    """
    if not data:
        return data

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
