from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import resolve_path


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def listdir(path: str = ".") -> str:
    """List entries in a directory under the workspace (name and type)."""
    target = resolve_path(path)
    if not target.is_dir():
        return f"error: not a directory: {path}"
    lines: list[str] = []
    for entry in sorted(target.iterdir(), key=lambda p: p.name):
        kind = "dir" if entry.is_dir() else "file"
        lines.append(f"{kind}\t{entry.name}")
    return "\n".join(lines) if lines else "(empty)"
