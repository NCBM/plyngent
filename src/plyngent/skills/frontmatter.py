"""YAML-subset frontmatter for ``SKILL.md`` files.

Skills are shared with other harnesses, which write a small YAML block above the
Markdown body. Supporting all of YAML would mean a dependency for a handful of
scalar fields, so this module reads the subset those files actually use:
top-level ``key: value`` pairs, quoted or plain scalars, booleans and integers,
inline ``[a, b]`` lists, block ``- item`` lists, and ``|`` / ``>`` block scalars.
A plain scalar folds the indented lines below it (YAML's multi-line form) and a
nested mapping is skipped rather than guessed at; anything structurally broken
raises :class:`FrontmatterError` so callers can fall back to "no metadata".
"""

from __future__ import annotations

import re

_FENCE = "---"
# Closing markers accepted by the format (``...`` is the YAML document end).
_CLOSERS = frozenset({_FENCE, "..."})
_BLOCK_SCALAR_MARKERS = frozenset({"|", "|-", "|+", ">", ">-", ">+"})
_TRUE = frozenset({"true", "yes", "on"})
_FALSE = frozenset({"false", "no", "off"})
_INT = re.compile(r"^-?\d+$")
_ITEM = "- "
# A quoted scalar needs at least its two quote characters.
_QUOTED_MIN_LEN = 2


class FrontmatterError(ValueError):
    """Raised when a frontmatter block is outside the supported subset."""


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """Return ``(metadata, body)`` for a ``SKILL.md`` document.

    No frontmatter (or an unterminated block) yields empty metadata and the whole
    text as the body.

    Raises:
        FrontmatterError: The block is outside the supported subset.
    """
    block, body = split_frontmatter(text)
    if block is None:
        return {}, body
    return parse_yaml_subset(block), body.removeprefix("\n")


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """Split *text* into its raw frontmatter block (or ``None``) and the body.

    The block must open with a ``---`` line of its own and close with ``---`` or
    ``...``; an unterminated block counts as no frontmatter, so the whole text
    stays readable as the body.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != _FENCE:
        return None, normalized
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() in _CLOSERS:
            return "\n".join(lines[1:index]), "\n".join(lines[index + 1 :])
    return None, normalized


def parse_yaml_subset(block: str) -> dict[str, object]:
    """Parse one frontmatter block into a flat mapping.

    Raises:
        FrontmatterError: A line is not a top-level ``key: value`` pair.
    """
    metadata: dict[str, object] = {}
    lines = block.split("\n")
    index = 0
    while index < len(lines):
        raw = lines[index]
        index += 1
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(_ITEM):
            msg = f"list item without a key: {raw!r}"
            raise FrontmatterError(msg)
        if _indent_of(raw) > 0:
            msg = f"unexpected indentation: {raw!r}"
            raise FrontmatterError(msg)
        key, sep, value = raw.partition(":")
        if not sep or not key.strip():
            msg = f"expected 'key: value', got {raw!r}"
            raise FrontmatterError(msg)
        name = key.strip()
        value = value.strip()
        consumed, index = _consume_indented(lines, index)
        if value in _BLOCK_SCALAR_MARKERS:
            metadata[name] = _join_block_scalar(consumed, folded=value.startswith(">"))
            continue
        if value:
            metadata[name] = _scalar(value, consumed)
            continue
        items = _list_items(consumed)
        if items is not None:
            metadata[name] = items
        # Empty value with nested mappings: nothing this subset needs to record.
    return metadata


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _consume_indented(lines: list[str], index: int) -> tuple[list[str], int]:
    """Consume lines indented deeper than the key line; return them and the next index."""
    consumed: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent_of(line) == 0:
            break
        consumed.append(line)
        index += 1
    return consumed, index


def _dedent(lines: list[str]) -> list[str]:
    """Strip the common indentation and trim blank edges."""
    indents = [_indent_of(line) for line in lines if line.strip()]
    if not indents:
        return []
    width = min(indents)
    dedented = [line[width:] if len(line) >= width else "" for line in lines]
    while dedented and not dedented[0].strip():
        _ = dedented.pop(0)
    while dedented and not dedented[-1].strip():
        _ = dedented.pop()
    return dedented


def _join_block_scalar(lines: list[str], *, folded: bool) -> str:
    """Join a ``|`` (literal) or ``>`` (folded) block scalar's content."""
    body = _dedent(lines)
    if not body:
        return ""
    return " ".join(line.strip() for line in body) if folded else "\n".join(body)


def _list_items(lines: list[str]) -> list[object] | None:
    """Parse a block ``- item`` list; ``None`` when *lines* is not one."""
    meaningful = [line for line in lines if line.strip()]
    if not meaningful or any(not line.strip().startswith(_ITEM) for line in meaningful):
        return None
    items: list[object] = []
    for line in meaningful:
        item = line.strip()[len(_ITEM) :].strip()
        if item:
            items.append(_coerce(item))
    return items


def _unquote(text: str) -> str | None:
    """Return the contents of a quoted scalar, or ``None`` when it is unquoted."""
    if len(text) >= _QUOTED_MIN_LEN and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return None


def _scalar(value: str, continuation: list[str]) -> object:
    """Parse one scalar; a plain scalar folds its indented continuation lines."""
    if value.startswith("[") and value.endswith("]"):
        return _inline_list(value)
    unquoted = _unquote(value)
    if unquoted is not None:
        return unquoted
    folded = [value]
    folded.extend(line.strip() for line in continuation if line.strip())
    return _coerce(" ".join(folded))


def _inline_list(value: str) -> list[object]:
    """Parse ``[a, "b", 3]`` (no nested collections in this subset)."""
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_coerce(part.strip()) for part in inner.split(",") if part.strip()]


def _coerce(text: str) -> object:
    """Coerce a plain scalar to bool / int / str (YAML's implicit typing)."""
    unquoted = _unquote(text)
    if unquoted is not None:
        return unquoted
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if _INT.match(text):
        return int(text)
    return text
