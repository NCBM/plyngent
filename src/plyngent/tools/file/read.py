from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import resolve_path


@tool
def read_file(path: str, *, offset: int = 0, limit: int | None = None) -> str:
    """Read a text file under the workspace.

    ``offset`` is 0-based line start; ``limit`` is max lines (None = rest of file).
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
    return "".join(lines[start:end])
