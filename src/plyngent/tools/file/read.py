from __future__ import annotations

from typing import cast

from plyngent.agent import ToolTag, mark_lineno_read, tool
from plyngent.tools.truncate_token import truncate_with_token
from plyngent.tools.workspace import resolve_path

_LINENO_WIDTH = 6


def _format_with_lineno(lines: list[str], *, start_lineno: int) -> str:
    """Prefix each line with a 1-based absolute line number (``edit_lineno`` style)."""
    out: list[str] = []
    for index, line in enumerate(lines):
        lineno = start_lineno + index
        # Strip keepends for the body; re-add a single newline after the prefix.
        body = line.rstrip("\r\n")
        out.append(f"{lineno:>{_LINENO_WIDTH}}|{body}\n")
    return "".join(out)


def line_range_label(text: str, start_char: int, end_char: int) -> str:
    """1-based inclusive line range covering ``text[start_char:end_char]``.

    Char offsets do not map 1:1 to lines; this converts a contiguous char range
    into the 1-based file line numbers it spans (offset is 0-based, so line 1
    starts at char 0).
    """
    begin = text.count("\n", 0, start_char) + 1
    last = max(start_char, end_char - 1)
    end = text.count("\n", 0, last) + 1
    return f"L{begin}-{end}"


def read_raw_text(path: str) -> tuple[str | None, str]:
    """(text, error) for a workspace file read; ``error`` is ``""`` on success.

    Shared by ``read_file`` and ``get_truncated`` so truncate-token char offsets
    always refer to the raw file text (never the ``L{begin}-{end}`` header).
    Missing paths and directories are distinguished for the caller.
    """
    target = resolve_path(path)
    if not target.exists():
        return None, f"error: file not found: {path}"
    if not target.is_file():
        return None, f"error: not a file: {path}"
    return target.read_text(encoding="utf-8", errors="replace"), ""


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def read_file(
    path: str,
    *,
    offset: int = 0,
    limit: int | None = None,
    with_lineno: bool = False,
    max_chars: int | None = None,
) -> str:
    """Read a text file under the workspace.

    ``offset`` is 0-based line start (0 = first line); ``limit`` is max lines
    (None = rest of file). When ``with_lineno`` is true, each line is prefixed
    with its 1-based file line number (``     N|…``), matching ``edit_lineno``
    numbering, and those lines are marked readable for ``edit_lineno`` this turn.

    ``max_chars`` caps the returned slice; a ``TRUNCATE_TOKEN`` is appended so
    ``get_truncated`` can continue reading the rest without a new request.

    The result starts with a 1-based inclusive line range ``L{begin}-{end}`` so
    the caller knows exactly which file lines were read (offset is 0-based).
    """
    target = resolve_path(path)
    text, err = read_raw_text(path)
    if err:
        return err
    text = cast("str", text)
    lines = text.splitlines(keepends=True)
    if offset < 0:
        return "error: offset must be >= 0"
    start = offset
    end = len(lines) if limit is None else min(len(lines), start + limit)
    if start >= len(lines):
        return ""
    slice_lines = lines[start:end]
    if with_lineno:
        mark_lineno_read(str(target), set(range(start + 1, end + 1)))
        body = _format_with_lineno(slice_lines, start_lineno=start + 1)
    else:
        body = "".join(slice_lines)
    if max_chars is not None and max_chars >= 1:
        char_start = len("".join(lines[:start]))
        body, _ = truncate_with_token(
            body,
            max_chars,
            kind="file",
            location=path,  # arg form: short token, resolves to the same file
            offset=char_start,
            limit=max_chars,
            total_len=len(text),
        )
    if not body:
        return ""
    if with_lineno:
        return body  # per-line numbers already show the range
    begin = start + 1
    end_line = start + len(slice_lines)
    return f"L{begin}-{end_line}\n{body}"
