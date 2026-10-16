from __future__ import annotations

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import AccessMode, resolve_path


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, writes=("path",))
async def write_file(path: str, content: str, *, overwrite: bool = False) -> str:
    """Write text content to a file under the workspace (creates parents).

    Refuses to replace an existing file: pass ``overwrite=true`` to replace it
    (the replacement is then confirmed by the human).
    """
    target = resolve_path(path, required=AccessMode.WRITE)
    if target.is_file() and not overwrite:
        return f"error: file exists: {path} (set overwrite=true to replace it)"
    target.parent.mkdir(parents=True, exist_ok=True)
    _ = target.write_text(content, encoding="utf-8")
    # The write changes the file's mtime, so edit_lineno's freshness check
    # rejects stale line numbers until the model re-reads with_lineno.
    return f"wrote {len(content)} characters to {path}"
