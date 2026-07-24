from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import (
    MAX_TEMPORARY_WORKSPACES,
    WorkspaceError,
    add_workspace_allowlist,
    list_workspace_allowlist,
    pop_owned_temporary_workspaces,
    remove_workspace_allowlist,
)

_DEFAULT_PREFIX = "ws"
_PREFIX_MAX_LEN = 32
_PREFIX_SAFE = re.compile(rf"^[A-Za-z0-9_-]{{1,{_PREFIX_MAX_LEN}}}$")


def _sanitize_prefix(prefix: str) -> str:
    token = (prefix or _DEFAULT_PREFIX).strip() or _DEFAULT_PREFIX
    # Replace unsafe characters so mkdtemp stays portable on Windows/POSIX.
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", token)
    cleaned = cleaned.strip("-_") or _DEFAULT_PREFIX
    if len(cleaned) > _PREFIX_MAX_LEN:
        cleaned = cleaned[:_PREFIX_MAX_LEN]
    if not _PREFIX_SAFE.match(cleaned):
        return _DEFAULT_PREFIX
    return cleaned


def _is_under_system_temp(path: Path) -> bool:
    """True if *path* is under the OS temporary directory (safe to rmtree)."""
    try:
        temp_root = Path(tempfile.gettempdir()).expanduser().resolve()
        resolved = path.expanduser().resolve()
        _ = resolved.relative_to(temp_root)
    except OSError, ValueError:
        return False
    return True


def cleanup_temporary_workspaces() -> int:
    """Remove directories created by :func:`new_temporary_workspace` (chat exit).

    Only deletes paths still under the system temp dir. Returns the number of
    directories removed.
    """
    removed = 0
    for path in pop_owned_temporary_workspaces():
        if not _is_under_system_temp(path):
            remove_workspace_allowlist(path)
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        remove_workspace_allowlist(path)
    return removed


def _create_temporary_workspace(prefix: str) -> str:
    if len(list_workspace_allowlist()) >= MAX_TEMPORARY_WORKSPACES:
        return f"error: too many temporary workspaces (max {MAX_TEMPORARY_WORKSPACES})"

    safe = _sanitize_prefix(prefix)
    try:
        # mkdtemp is cross-platform; prefix must end with - for readability.
        path_str = tempfile.mkdtemp(prefix=f"plyngent-{safe}-")
        path = Path(path_str).resolve()
    except OSError as exc:
        return f"error: failed to create temporary workspace: {exc}"

    try:
        _ = add_workspace_allowlist(path, owned=True)
    except WorkspaceError as exc:
        shutil.rmtree(path, ignore_errors=True)
        return f"error: {exc}"

    return (
        f"temporary_workspace={path}\n"
        "note: project workspace unchanged; use this absolute path for tools; "
        "removed when chat exits"
    )


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def new_temporary_workspace(prefix: str = "ws") -> str:
    """Create a scratch directory under the system temp dir and allow tool paths in it.

    The project workspace is unchanged: relative paths still resolve there.
    Use the returned **absolute** path for file/process tools. Temporary
    workspaces are deleted when this chat process exits (not on each turn).

    Cross-platform (uses ``tempfile``; typically ``/tmp`` on POSIX, ``%TEMP%``
    on Windows). At most 16 concurrent temporary workspaces per process.
    """
    import asyncio

    return await asyncio.to_thread(_create_temporary_workspace, prefix)
