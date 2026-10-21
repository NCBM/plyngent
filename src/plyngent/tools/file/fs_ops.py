from __future__ import annotations

import shutil
from pathlib import Path

from plyngent.agent import ToolTag, invalidate_lineno_read, tool
from plyngent.tools.trash import move_to_trash, under_system_temp
from plyngent.tools.workspace import AccessMode, WorkspaceError, get_workspace_root, resolve_path


def _kind(path: Path) -> str:
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    if path.is_symlink():
        return "symlink"
    return "path"


def _remove_existing(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _resolve_pair(src: str, dst: str) -> tuple[Path, Path, Path] | str:
    try:
        source = resolve_path(src, required=AccessMode.READ)
        dest = resolve_path(dst, required=AccessMode.WRITE)
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"
    return source, dest, root


def _display(path: Path, root: Path) -> str:
    """Workspace-relative display path, falling back to absolute outside the root."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _prepare_dest(source: Path, dest: Path, *, overwrite: bool, dst_label: str) -> Path | str:
    """Resolve copy/move destination; return error string on conflict."""
    if source.is_file() and dest.is_dir():
        # ``dest`` may be outside the primary root (temp workspace / grant), so
        # resolve the child directly instead of relativizing against the root.
        dest = resolve_path(str(dest / source.name), required=AccessMode.WRITE)
    if dest.exists() or dest.is_symlink():
        if not overwrite:
            return f"error: destination exists: {dst_label} (set overwrite=true)"
        _remove_existing(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return dest


def _copy_or_move_validated(
    source: Path,
    dest: Path,
    root: Path,
    *,
    src: str,
    dst: str,
    overwrite: bool,
    action: str,
) -> str:
    prepared = _prepare_dest(source, dest, overwrite=overwrite, dst_label=dst)
    if isinstance(prepared, str):
        return prepared
    dest = prepared
    try:
        if action == "copy":
            if source.is_dir() and not source.is_symlink():
                _ = shutil.copytree(source, dest, symlinks=True)
                label = "directory"
            else:
                _ = shutil.copy2(source, dest, follow_symlinks=False)
                label = _kind(source)
            # A copy does not change the source, but the destination content is
            # new. copy2/copytree preserve the source mtime, so the freshness
            # check alone could miss the change — invalidate explicitly.
            invalidate_lineno_read(str(dest))
            return f"copied {label} {src} -> {_display(dest, root)}"
        moved = Path(shutil.move(str(source), str(dest))).resolve()
        # shutil.move preserves mtime too; drop read state for both paths.
        invalidate_lineno_read(str(source))
        invalidate_lineno_read(str(moved))
        return f"moved {_kind(moved)} {src} -> {_display(moved, root)}"
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: {action} failed: {exc}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, writes=("dst",))
async def copy_path(src: str, dst: str, *, overwrite: bool = False) -> str:
    """Copy a file or directory under the workspace (``shutil.copy2`` / ``copytree``).

    ``dst`` parent directories are created as needed. If ``dst`` exists, set
    ``overwrite=true`` to replace it.
    """
    pair = _resolve_pair(src, dst)
    if isinstance(pair, str):
        return pair
    source, dest, root = pair
    if not source.exists() and not source.is_symlink():
        return f"error: source does not exist: {src}"
    return _copy_or_move_validated(source, dest, root, src=src, dst=dst, overwrite=overwrite, action="copy")


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, writes=("src", "dst"))
async def move_path(src: str, dst: str, *, overwrite: bool = False) -> str:
    """Move/rename a file or directory under the workspace (``shutil.move``)."""
    pair = _resolve_pair(src, dst)
    if isinstance(pair, str):
        return pair
    source, dest, root = pair
    if not source.exists() and not source.is_symlink():
        return f"error: source does not exist: {src}"
    if source == root:
        return "error: cannot move the workspace root"
    return _copy_or_move_validated(source, dest, root, src=src, dst=dst, overwrite=overwrite, action="move")


def _discard(target: Path, path: str, *, kind: str, permanent: bool, detail: str = "") -> str:
    """Move *target* to the system trash (default), or delete it for good."""
    if permanent:
        _remove_existing(target)
        invalidate_lineno_read(str(target))
        return f"deleted {kind} {path}{detail}"
    error = move_to_trash(target)
    if error is not None:
        return error
    invalidate_lineno_read(str(target))
    return f"deleted {kind} {path} (moved to system trash)"


def _delete_target(target: Path, path: str, *, recursive: bool, permanent: bool) -> str:
    if target.is_dir() and not target.is_symlink():
        if recursive:
            return _discard(target, path, kind="directory", permanent=permanent, detail=" (recursive)")
        try:
            _ = next(target.iterdir())
        except StopIteration:
            return _discard(target, path, kind="empty directory", permanent=permanent)
        return f"error: directory not empty: {path} (set recursive=true)"
    if target.is_symlink() or target.is_file():
        return _discard(target, path, kind=_kind(target), permanent=permanent)
    return f"error: unsupported path type: {path}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO, writes=("path",))
async def delete_path(path: str, *, recursive: bool = False, permanent: bool = False) -> str:
    """Delete a file or directory under the workspace.

    The path goes to the system trash (the desktop's own trash / Recycle Bin), so
    a deletion can be undone there; ``permanent=true`` removes it for good
    instead. Files and empty directories are removed always, while non-empty
    directories require ``recursive=true`` (uses ``shutil.rmtree``). The
    workspace root cannot be deleted. When the trash is unavailable for this
    path, nothing is deleted and the error says how to delete on purpose.

    Paths under the OS temporary directory are deleted for good either way: they
    are scratch space, not something to restore.
    """
    try:
        target = resolve_path(path, required=AccessMode.WRITE)
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"

    if not target.exists() and not target.is_symlink():
        return f"error: path does not exist: {path}"
    if target == root:
        return "error: cannot delete the workspace root"
    try:
        return _delete_target(
            target,
            path,
            recursive=recursive,
            permanent=permanent or under_system_temp(target),
        )
    except OSError as exc:
        return f"error: delete failed: {exc}"
