"""The frontmatter of a ``SKILL.md``: the small YAML block every harness writes above the body.

Skills are shared with other programs, and they all write their block as YAML — so this reads YAML,
but only the part those files actually use: top-level ``key: value`` pairs, plain and quoted scalars,
booleans and integers, inline ``[a, b]`` lists, block ``- item`` lists, and ``|`` / ``>`` block scalars.
A plain scalar folds the indented lines under it, which is YAML's own multi-line form.

**A whole YAML implementation is a dependency for six fields, and a guess is worse than a refusal.**
So the subset is a contract: a block outside it raises :class:`FrontmatterError` and the caller falls
back to "no metadata, the directory's name and the first paragraph" — a third-party skill is then
listed with a note saying it was not parsed, rather than silently repaired into something it is not.
Nested mappings are the one construct skipped rather than refused: they carry no field this kernel
reads, and refusing the whole block over one would cost a skill its description as well.
"""

from __future__ import annotations

import re

#: The line that opens and closes a frontmatter block.
_FENCE = "---"
#: What may close one: the fence itself, or YAML's document end.
_CLOSERS = frozenset({_FENCE, "..."})
#: The block-scalar markers, which say whether the lines under them keep their breaks.
_BLOCK_SCALAR_MARKERS = frozenset({"|", "|-", "|+", ">-", ">", ">+"})
_TRUE = frozenset({"true", "yes", "on"})
_FALSE = frozenset({"false", "no", "off"})
_INT = re.compile(r"^-?\d+$")
_ITEM = "- "
#: A quoted scalar has at least its two quote characters, so shorter text is never quoted.
_QUOTED_MIN_LEN = 2


class FrontmatterError(ValueError):
    """The block is outside the subset this module reads, so the skill has no metadata."""


def parse_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """The metadata and the body of a ``SKILL.md``.

    No frontmatter — or an unterminated block, which is a file somebody was still writing — yields no
    metadata and the whole text as the body, because the body is the part worth reading either way.

    Raises:
        FrontmatterError: The block is outside the subset.
    """
    block, body = split_frontmatter(text)
    if block is None:
        return {}, body
    return parse_yaml_subset(block), body.removeprefix("\n")


def split_frontmatter(text: str) -> tuple[str | None, str]:
    """*text* as its raw frontmatter block (or ``None``) and its body.

    The block is a ``---`` line of its own, closed by ``---`` or ``...``; anything else is a body that
    happens to begin with a horizontal rule.
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
    """One frontmatter block as a flat mapping.

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
        key, separator, value = raw.partition(":")
        if not separator or not key.strip():
            msg = f"expected 'key: value', got {raw!r}"
            raise FrontmatterError(msg)
        name = key.strip()
        rest = value.strip()
        consumed, index = _consume_indented(lines, index)
        if rest in _BLOCK_SCALAR_MARKERS:
            metadata[name] = _join_block_scalar(consumed, folded=rest.startswith(">"))
        elif rest:
            metadata[name] = _scalar(rest, consumed)
        else:
            items = _list_items(consumed)
            if items is not None:
                metadata[name] = items
            # An empty value with nested mappings under it: nothing this subset records.
    return metadata


def _indent_of(line: str) -> int:
    """How many spaces a line is indented by."""
    return len(line) - len(line.lstrip(" "))


def _consume_indented(lines: list[str], index: int) -> tuple[list[str], int]:
    """The lines indented under the key at ``index - 1``, and where the next key starts."""
    consumed: list[str] = []
    while index < len(lines):
        line = lines[index]
        if line.strip() and _indent_of(line) == 0:
            break
        consumed.append(line)
        index += 1
    return consumed, index


def _dedent(lines: list[str]) -> list[str]:
    """The common indentation taken off, and the blank edges trimmed."""
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
    """A ``|`` block kept as lines, or a ``>`` block folded into one."""
    body = _dedent(lines)
    if not body:
        return ""
    return " ".join(line.strip() for line in body) if folded else "\n".join(body)


def _list_items(lines: list[str]) -> list[object] | None:
    """A block ``- item`` list, or ``None`` when these lines are not one."""
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
    """What a quoted scalar says, or ``None`` when it is not quoted."""
    if len(text) >= _QUOTED_MIN_LEN and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return None


def _scalar(value: str, continuation: list[str]) -> object:
    """One scalar, with a plain scalar folding the indented lines under it."""
    if value.startswith("[") and value.endswith("]"):
        return _inline_list(value)
    unquoted = _unquote(value)
    if unquoted is not None:
        return unquoted
    folded = [value]
    folded.extend(line.strip() for line in continuation if line.strip())
    return _coerce(" ".join(folded))


def _inline_list(value: str) -> list[object]:
    """``[a, "b", 3]`` — no nested collections in this subset."""
    inner = value[1:-1].strip()
    if not inner:
        return []
    return [_coerce(part.strip()) for part in inner.split(",") if part.strip()]


def _coerce(text: str) -> object:
    """A plain scalar as YAML's implicit typing reads it: bool, integer, or the text itself."""
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
