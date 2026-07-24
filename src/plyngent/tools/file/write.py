from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import resolve_path


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def write_file(path: str, content: str) -> str:
    """Write text content to a file under the workspace (creates parents)."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} characters to {path}"
