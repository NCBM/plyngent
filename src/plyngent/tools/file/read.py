from __future__ import annotations

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

    ``offset`` is 0-based line start; ``limit`` is max lines (None = rest of file).
    When ``with_lineno`` is true, each line is prefixed with its 1-based file line
    number (``     N|…``), matching ``edit_lineno`` numbering, and those lines are
    marked readable for ``edit_lineno`` this turn.

    ``max_chars`` caps the returned slice; a ``TRUNCATE_TOKEN`` is appended so
    ``get_truncated`` can continue reading the rest without a new request.
    """
    target = resolve_path(path)
    if not target.is_file():
        return f"error: not a file: {path}"
    text = target.read_text(encoding="utf-8", errors="replace")
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
        out = _format_with_lineno(slice_lines, start_lineno=start + 1)
    else:
        out = "".join(slice_lines)
    if max_chars is not None and max_chars >= 1:
        char_start = len("".join(lines[:start]))
        out, _ = truncate_with_token(
            out,
            max_chars,
            kind="file",
            location=path,  # arg form: short token, resolves to the same file
            offset=char_start,
            limit=max_chars,
            total_len=len(text),
        )
    return out
