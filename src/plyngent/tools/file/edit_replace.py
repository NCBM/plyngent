from __future__ import annotations

from plyngent.agent import tool
from plyngent.tools.workspace import resolve_path


@tool
def edit_replace(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of ``old_string`` with ``new_string`` in a file."""
    if not old_string:
        return "error: old_string must not be empty"
    target = resolve_path(path)
    if not target.is_file():
        return f"error: not a file: {path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    if old_string not in text:
        return "error: old_string not found in file"
    updated = text.replace(old_string, new_string, 1)
    _ = target.write_text(updated, encoding="utf-8")
    return f"replaced first occurrence in {path}"
