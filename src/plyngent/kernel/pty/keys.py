"""What ``<key>`` markers mean: the bytes a terminal would send, and nothing typed by accident.

A model driving an interactive program has to be able to say "press Ctrl-C" without saying it in
characters it cannot type. So a write may carry markers — ``<enter>``, ``<c-up>``, ``<f5>`` — and this
module turns them into the sequence xterm would send. Two rules keep that from being a foot-gun:

* **A marker that names nothing is an error, not text.** Typing ``<c-uop>`` into a shell prompt because
  of a typo is worse than being told the name does not exist.
* **A ``<`` that opens nothing stays a ``<``.** ``sort < file`` is ordinary input and must not need
  escaping; only a short, whitespace-free ``<…>`` is read as a marker, and ``<<`` is how a literal one
  is written.

Sanitizing output used to live beside this and no longer does: making child bytes safe to *read* is the
renderer's business (:mod:`plyngent.kernel.render`), and this module only ever produces bytes to write.
"""

from __future__ import annotations

import re

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

#: Keys a modifier applies to, as the xterm sequence with the modifier slot left open.
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

#: The longest marker content to look for, so a stray ``<`` stays a literal ``<``.
_MAX_MARKER = 32

_HEX_BYTE = re.compile(r"\\x([0-9A-Fa-f]{2})")
_HEX_CODE_POINT = re.compile(r"\\u([0-9A-Fa-f]{4})")
_SIMPLE: dict[str, str] = {
    r"\n": "\n",
    r"\r": "\r",
    r"\t": "\t",
    r"\e": "\x1b",
    r"\E": "\x1b",
    r"\0": "\x00",
}
_SIMPLE_LENGTH = 2


class KeySyntaxError(ValueError):
    """A ``<...>`` marker that names no key, or repeats it wrongly."""


def decode_write_data(data: str) -> str:
    """Expand ``<key>`` markers and escapes into the bytes to write to a terminal.

    Markers are written between angle brackets: the keys a terminal understands, with a modifier in
    front where a key has one (``<c-up>``, ``<s-home>``, ``<c-s-f5>``) — on a plain character Ctrl is
    its control byte and Alt an ESC before it. ``<up*3>`` repeats a key, and ``<<`` is a literal ``<``.

    Escapes are literal characters in ``data``: ``\\n`` ``\\r`` ``\\t`` ``\\e``/``\\E`` (ESC) ``\\0``,
    ``\\xHH`` for one byte and ``\\uHHHH`` for one code point. Anything else is sent as it is.

    Raises :class:`KeySyntaxError` for a marker that names nothing: a typo must be reported rather than
    typed into somebody's shell prompt.
    """
    if not data:
        return data

    out: list[str] = []
    index = 0
    end = len(data)
    while index < end:
        if data[index] == "<":
            if data.startswith("<<", index):
                out.append("<")
                index += 2
                continue
            marker = _marker_at(data, index)
            if marker is not None:
                content, index = marker
                out.append(_expansion(content))
                continue
            out.append("<")
            index += 1
            continue
        rest = data[index:]
        if (hex_byte := _HEX_BYTE.match(rest)) is not None:
            out.append(chr(int(hex_byte.group(1), 16)))
            index += hex_byte.end()
            continue
        if (hex_code := _HEX_CODE_POINT.match(rest)) is not None:
            out.append(chr(int(hex_code.group(1), 16)))
            index += hex_code.end()
            continue
        pair = rest[:_SIMPLE_LENGTH]
        if len(pair) == _SIMPLE_LENGTH and pair in _SIMPLE:
            out.append(_SIMPLE[pair])
            index += _SIMPLE_LENGTH
            continue
        out.append(data[index])
        index += 1
    return "".join(out)


def _marker_at(data: str, index: int) -> tuple[str, int] | None:
    """The marker starting at *index*, as ``(content, next index)``, or ``None``.

    A ``<`` that is not followed by a short, whitespace-free ``>`` pair opens nothing and stays a
    literal: ``sort < file`` is ordinary input, and requiring an escape for it would be a trap.
    """
    stop = data.find(">", index + 1, index + 1 + _MAX_MARKER)
    if stop <= index + 1:
        return None
    content = data[index + 1 : stop]
    if any(char.isspace() or char == "<" for char in content):
        return None
    return content, stop + 1


def _expansion(content: str) -> str:
    """The bytes for the inside of one marker, honouring a ``*N`` repeat."""
    name, _, count = content.partition("*")
    if not count:
        return _key_bytes(name)
    if not count.isdigit() or int(count) < 1:
        msg = f"bad repeat in <{content}> (write <up*3>)"
        raise KeySyntaxError(msg)
    return _key_bytes(name) * int(count)


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
    """One key under Ctrl, Alt or Shift, spelled the way xterm spells it."""
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


def _control(char: str) -> str:
    """The C0 byte for Ctrl and one character."""
    if len(char) != 1:
        msg = f"ctrl takes one character, not {char!r} (write <c-a>)"
        raise KeySyntaxError(msg)
    upper = char.upper()
    if not ("A" <= upper <= "Z" or char in _CONTROL_CHARS):
        msg = f"no control byte for <c-{char}>"
        raise KeySyntaxError(msg)
    return chr(ord(upper) & 0x1F)
