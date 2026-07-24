from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import resolve_path


def _count_non_overlapping(text: str, needle: str) -> int:
    """Count left-to-right non-overlapping matches (same as ``str.replace``)."""
    if not needle:
        return 0
    n = 0
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return n
        n += 1
        start = index + len(needle)


def _success_message(path: str, *, replaced: int, found: int) -> str:
    remaining = found - replaced
    unit = "occurrence" if replaced == 1 else "occurrences"
    if remaining == 0:
        if found == 1:
            return f"replaced 1 occurrence in {path}"
        return f"replaced {replaced} {unit} in {path} (all {found} matches)"
    return (
        f"replaced {replaced} {unit} in {path} "
        f"({replaced} of {found} matches; {remaining} remain — "
        f"raise max_replaces or use a more specific old_string)"
    )


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def edit_replace(path: str, old_string: str, new_string: str, max_replaces: int = 1) -> str:
    """Replace occurrences of ``old_string`` with ``new_string`` in a file.

    Replaces left-to-right, non-overlapping. Default ``max_replaces=1`` (first match
    only). Raise ``max_replaces`` to change multiple identical hits; use a more
    specific ``old_string`` when you need a particular occurrence.
    """
    if not old_string:
        return "error: old_string must not be empty"
    if max_replaces < 1:
        return "error: max_replaces must be >= 1"
    target = resolve_path(path)
    if not target.is_file():
        return f"error: not a file: {path}"
    text = target.read_text(encoding="utf-8", errors="replace")
    found = _count_non_overlapping(text, old_string)
    if found == 0:
        return "error: old_string not found in file"
    n = min(max_replaces, found)
    updated = text.replace(old_string, new_string, n)
    _ = target.write_text(updated, encoding="utf-8")
    return _success_message(path, replaced=n, found=found)
