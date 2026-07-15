from __future__ import annotations

from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from collections.abc import Callable

_TRIPLE = '"""'


def parse_triple_quote_line(line: str) -> tuple[str, bool] | None:
    """If ``line`` starts a ``\"\"\"`` block, return ``(content_so_far, complete)``.

    ``complete`` is True when the closing marker is on the same line
    (e.g. ``\"\"\"hello\"\"\"`` → ``(\"hello\", True)``).
    """
    stripped = line.strip()
    if not stripped.startswith(_TRIPLE):
        return None
    rest = stripped[len(_TRIPLE) :]
    if rest.endswith(_TRIPLE) and len(rest) >= len(_TRIPLE):
        inner = rest[: -len(_TRIPLE)]
        return inner, True
    return rest, False


def finish_triple_quote_line(line: str) -> tuple[str, bool]:
    """Parse a continuation line. Returns ``(content_piece, is_closer)``."""
    stripped = line.strip()
    if stripped == _TRIPLE:
        return "", True
    # Allow closing as trailing """ on a content line.
    if line.rstrip().endswith(_TRIPLE):
        without = line.rstrip()[: -len(_TRIPLE)]
        # Don't treat a line that is only """ as content (handled above).
        return without.rstrip("\n"), True
    return line, False


def assemble_multiline(
    opening_line: str,
    *,
    read_line: Callable[[], str],
    echo: Callable[[str], None] | None = None,
) -> str | None:
    """Read a full triple-quoted message. Empty body → ``None`` (cancel)."""
    parsed = parse_triple_quote_line(opening_line)
    if parsed is None:
        text = opening_line.strip()
        return text or None
    first, complete = parsed
    if complete:
        return first if first.strip() else None

    parts: list[str] = []
    if first:
        parts.append(first)
    if echo is not None:
        echo(f"(multiline; end with {_TRIPLE})")
    while True:
        try:
            line = read_line()
        except EOFError, KeyboardInterrupt:
            if echo is not None:
                echo("")
                echo("cancelled")
            return None
        piece, done = finish_triple_quote_line(line)
        if done:
            if piece:
                parts.append(piece)
            break
        parts.append(piece)
    text = "\n".join(parts)
    return text if text.strip() else None


def _default_prompt() -> str:
    return input("> ")


def _default_cont() -> str:
    return input("... ")


def _default_echo(message: str) -> None:
    click.echo(message)


def read_repl_entry(
    *,
    read_line: Callable[[], str] | None = None,
    echo: Callable[[str], None] | None = None,
) -> str | None:
    """Read one REPL entry (slash line, single line, or multiline).

    Returns:
        ``None`` — empty line or cancelled multiline (caller should re-prompt).
        ``str`` starting with ``/`` — slash command line (stripped).
        other ``str`` — user message text (may contain newlines).
    """
    _read = read_line if read_line is not None else _default_prompt
    _echo = echo if echo is not None else _default_echo

    try:
        first = _read()
    except EOFError:
        raise
    except KeyboardInterrupt:
        _echo("")
        return None

    stripped = first.strip()
    if not stripped:
        return None
    if stripped.startswith("/") and parse_triple_quote_line(stripped) is None:
        return stripped

    if parse_triple_quote_line(first) is not None:
        cont_read = read_line if read_line is not None else _default_cont
        return assemble_multiline(first, read_line=cont_read, echo=_echo)

    return stripped
