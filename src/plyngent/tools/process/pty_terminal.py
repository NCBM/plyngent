"""PTY payload helpers for agent tools.

Child PTY I/O stays on the master FD. Tool results are printed on the *user*
TTY by the CLI, so any CSI/control bytes in ``read_pty`` data would reprogram
the host terminal. We sanitize on the tool boundary instead of resetting the
host after close (which flashes the screen on every chat exit).
"""

from __future__ import annotations

import re

# Escapes and ``<name>`` key markers in write_pty_keys data.
_HEX_BYTE = re.compile(r"\\x([0-9A-Fa-f]{2})")
_HEX_U = re.compile(r"\\u([0-9A-Fa-f]{4})")
_SIMPLE = {
    r"\n": "\n",
    r"\r": "\r",
    r"\t": "\t",
    r"\e": "\x1b",
    r"\E": "\x1b",
    r"\0": "\x00",
}
_SIMPLE_ESCAPE_LEN = 2

#: ``<name>`` markers: the keys a terminal understands, and the bytes each sends.
_KEY_NAMES: dict[str, str] = {
    "esc": "\x1b",
    "enter": "\r",
    "tab": "\t",
    "s-tab": "\x1b[Z",
    "space": " ",
    "bs": "\x7f",
    "del": "\x1b[3~",
    "ins": "\x1b[2~",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "left": "\x1b[D",
    "right": "\x1b[C",
    "pgup": "\x1b[5~",
    "pgdn": "\x1b[6~",
    "f1": "\x1bOP",
    "f2": "\x1bOQ",
    "f3": "\x1bOR",
    "f4": "\x1bOS",
    "f5": "\x1b[15~",
    "f6": "\x1b[17~",
    "f7": "\x1b[18~",
    "f8": "\x1b[19~",
    "f9": "\x1b[20~",
    "f10": "\x1b[21~",
    "f11": "\x1b[23~",
    "f12": "\x1b[24~",
}

#: The longer spellings, so a marker can be written the way it is usually named.
_KEY_ALIASES: dict[str, str] = {
    "escape": "esc",
    "return": "enter",
    "backspace": "bs",
    "delete": "del",
    "insert": "ins",
    "pageup": "pgup",
    "pagedown": "pgdn",
}

#: Keys a modifier applies to, as the xterm sequence with the modifier slot.
_MODIFIED: dict[str, str] = {
    "up": "\x1b[1;{m}A",
    "down": "\x1b[1;{m}B",
    "right": "\x1b[1;{m}C",
    "left": "\x1b[1;{m}D",
    "home": "\x1b[1;{m}H",
    "end": "\x1b[1;{m}F",
    "ins": "\x1b[2;{m}~",
    "del": "\x1b[3;{m}~",
    "pgup": "\x1b[5;{m}~",
    "pgdn": "\x1b[6;{m}~",
    "f1": "\x1b[1;{m}P",
    "f2": "\x1b[1;{m}Q",
    "f3": "\x1b[1;{m}R",
    "f4": "\x1b[1;{m}S",
    "f5": "\x1b[15;{m}~",
    "f6": "\x1b[17;{m}~",
    "f7": "\x1b[18;{m}~",
    "f8": "\x1b[19;{m}~",
    "f9": "\x1b[20;{m}~",
    "f10": "\x1b[21;{m}~",
    "f11": "\x1b[23;{m}~",
    "f12": "\x1b[24;{m}~",
}

#: Modifier prefixes a marker may carry: Ctrl, Alt (either spelling), Shift.
_MODIFIERS = frozenset({"c", "a", "m", "s"})

#: What a Ctrl may be combined with beyond A-Z: the rest of the C0 controls.
_CONTROL_CHARS = "@[\\]^_"
#: Longest marker content to look for, so a stray ``<`` stays a literal ``<``.
_MAX_MARKER = 32

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


class KeySyntaxError(ValueError):
    """A ``<...>`` marker that names no key, or repeats it wrongly."""


def _control(char: str) -> str:
    """The C0 byte for a Ctrl and one character."""
    if len(char) != 1:
        msg = f"ctrl takes one character, not {char!r} (write <c-a>)"
        raise KeySyntaxError(msg)
    upper = char.upper()
    if not ("A" <= upper <= "Z" or char in _CONTROL_CHARS):
        msg = f"no control byte for <c-{char}>"
        raise KeySyntaxError(msg)
    return chr(ord(upper) & 0x1F)


def _canonical(name: str) -> str:
    lowered = name.lower()
    return _KEY_ALIASES.get(lowered, lowered)


def _key_bytes(name: str) -> str:
    """What one ``<name>`` marker sends, modifiers and all."""
    known = _KEY_NAMES.get(_canonical(name))
    if known is not None:
        return known
    *prefixes, base = name.split("-")
    modifiers = {prefix.lower() for prefix in prefixes}
    if prefixes and base and modifiers <= _MODIFIERS:
        return _modified(name, base, modifiers)
    msg = f"unknown key <{name}> (see the write_pty_keys docs for the names)"
    raise KeySyntaxError(msg)


def _modified(name: str, base: str, modifiers: set[str]) -> str:
    """One key under Ctrl / Alt / Shift, spelled the way xterm spells it."""
    ctrl = "c" in modifiers
    alt = bool(modifiers & {"a", "m"})
    shift = "s" in modifiers
    template = _MODIFIED.get(_canonical(base))
    if template is not None:
        # xterm's modifier parameter: 1 + shift(1) + alt(2) + ctrl(4).
        return template.format(m=1 + shift * 1 + alt * 2 + ctrl * 4)
    if ctrl and _canonical(base) == "space":
        return "\x00"  # Ctrl+Space is NUL, not a CSI sequence
    if len(base) == 1:
        if shift:
            msg = f"<s-{base}> is just {base.upper()!r} — drop the modifier"
            raise KeySyntaxError(msg)
        return ("\x1b" if alt else "") + (_control(base) if ctrl else base)
    known = _KEY_NAMES.get(_canonical(base))
    if known is not None and not (ctrl or shift):
        return "\x1b" + known  # Alt alone: an ESC before the key, as terminals send it
    msg = f"no sequence for <{name}> (write the raw bytes with \\x1b)"
    raise KeySyntaxError(msg)


def _expansion(content: str) -> str:
    """The bytes for the inside of one marker, honouring a ``*N`` repeat."""
    name, _, count = content.partition("*")
    if not count:
        return _key_bytes(name)
    if not count.isdigit() or int(count) < 1:
        msg = f"bad repeat in <{content}> (write <up*3>)"
        raise KeySyntaxError(msg)
    return _key_bytes(name) * int(count)


def _marker_at(data: str, index: int) -> tuple[str, int] | None:
    """The marker starting at *index*, as ``(content, next index)``, or ``None``.

    A ``<`` that is not followed by a short, whitespace-free ``>`` pair stays a
    literal ``<`` — ordinary input like ``sort < file`` must not need escaping.
    """
    end = data.find(">", index + 1, index + 1 + _MAX_MARKER)
    if end <= index + 1:
        return None
    content = data[index + 1 : end]
    if any(char.isspace() or char == "<" for char in content):
        return None
    return content, end + 1


def decode_write_data(data: str) -> str:
    """Expand ``<key>`` markers and escapes into raw PTY input (``write_pty_keys``).

    Markers, written between angle brackets:

    - ``<esc>`` ``<enter>`` ``<tab>`` ``<s-tab>`` ``<space>`` ``<bs>`` ``<del>``
      ``<ins>`` ``<home>`` ``<end>`` ``<up>`` ``<down>`` ``<left>`` ``<right>``
      ``<pgup>`` ``<pgdn>`` ``<f1>`` … ``<f12>``
    - a modifier in front, alone or combined: ``<c-up>`` ``<s-home>``
      ``<m-tab>`` ``<c-s-f5>`` — the xterm sequence where a key has one. On a
      plain character, Ctrl is its control byte (``<c-a>``, ``<c-[>``) and Alt
      is an ESC before it (``<m-x>``)
    - ``<up*3>`` repeats a key three times, ``<<`` is a literal ``<``

    An unknown name raises :class:`KeySyntaxError` rather than typing the marker
    into the terminal. Anything that is not a marker or an escape is literal.

    Escapes (literal characters in ``data``):

    - ``\\\\n`` ``\\\\r`` ``\\\\t`` ``\\\\e`` / ``\\\\E`` (ESC) ``\\\\0``
    - ``\\\\xHH`` byte, ``\\\\uHHHH`` Unicode code point
    """
    if not data:
        return data

    out: list[str] = []
    i = 0
    n = len(data)
    while i < n:
        if data[i] == "<":
            if data.startswith("<<", i):
                out.append("<")  # the escape for a literal one
                i += 2
                continue
            marker = _marker_at(data, i)
            if marker is not None:
                content, i = marker
                out.append(_expansion(content))
                continue
            out.append("<")  # a `<` that opens nothing
            i += 1
            continue
        rest = data[i:]
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
