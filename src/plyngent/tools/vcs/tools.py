from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, get_workspace_root

from .detect import detect_vcs

if TYPE_CHECKING:
    from .backend import VcsBackend


def _backend_or_error() -> VcsBackend | str:
    try:
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"
    backend = detect_vcs(root)
    if backend is None:
        return "error: no supported VCS detected under workspace (currently: git; other systems can register detectors)"
    return backend


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def vcs_kind() -> str:
    """Return the detected VCS kind under the workspace (e.g. ``git``), or an error."""
    backend = _backend_or_error()
    if isinstance(backend, str):
        return backend
    return backend.kind


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def vcs_status() -> str:
    """Show working-tree status for the detected VCS (read-only)."""
    backend = _backend_or_error()
    if isinstance(backend, str):
        return backend
    return backend.status()


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def vcs_diff(path: str = "", *, staged: bool = False) -> str:
    """Show a unified diff for the detected VCS (read-only).

    ``path`` is optional and relative to the workspace. ``staged=true`` is
    honored by git (index vs HEAD); other backends may ignore it.
    """
    backend = _backend_or_error()
    if isinstance(backend, str):
        return backend
    rel = path.strip() or None
    return backend.diff(staged=staged, path=rel)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def vcs_log(limit: int = 10) -> str:
    """Show recent commits for the detected VCS (read-only)."""
    if limit < 1:
        return "error: limit must be >= 1"
    backend = _backend_or_error()
    if isinstance(backend, str):
        return backend
    return backend.log(limit=limit)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def vcs_branch() -> str:
    """Show the current branch / named head for the detected VCS (read-only)."""
    backend = _backend_or_error()
    if isinstance(backend, str):
        return backend
    return backend.branch()


VCS_TOOLS = [
    vcs_kind,
    vcs_status,
    vcs_diff,
    vcs_log,
    vcs_branch,
]
