from __future__ import annotations

import shutil
from pathlib import Path

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, get_workspace_root, resolve_path


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
        source = resolve_path(src)
        dest = resolve_path(dst)
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"
    return source, dest, root


def _prepare_dest(source: Path, dest: Path, root: Path, *, overwrite: bool, dst_label: str) -> Path | str:
    """Resolve copy/move destination; return error string on conflict."""
    if source.is_file() and dest.is_dir():
        dest = resolve_path(str((dest / source.name).relative_to(root)))
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
    prepared = _prepare_dest(source, dest, root, overwrite=overwrite, dst_label=dst)
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
            return f"copied {label} {src} -> {dest.relative_to(root)}"
        moved = Path(shutil.move(str(source), str(dest))).resolve()
        return f"moved {_kind(moved)} {src} -> {moved.relative_to(root)}"
    except WorkspaceError as exc:
        return f"error: {exc}"
    except OSError as exc:
        return f"error: {action} failed: {exc}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
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


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
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


def _delete_directory(target: Path, path: str, *, recursive: bool) -> str:
    if recursive:
        shutil.rmtree(target)
        return f"deleted directory {path} (recursive)"
    try:
        _ = next(target.iterdir())
    except StopIteration:
        target.rmdir()
        return f"deleted empty directory {path}"
    return f"error: directory not empty: {path} (set recursive=true)"


def _delete_target(target: Path, path: str, *, recursive: bool) -> str:
    if target.is_symlink() or target.is_file():
        kind = _kind(target)
        target.unlink()
        return f"deleted {kind} {path}"
    if target.is_dir():
        return _delete_directory(target, path, recursive=recursive)
    return f"error: unsupported path type: {path}"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.YOLO)
async def delete_path(path: str, *, recursive: bool = False) -> str:
    """Delete a file or directory under the workspace.

    Files and empty directories are removed always. Non-empty directories require
    ``recursive=true`` (uses ``shutil.rmtree``). The workspace root cannot be deleted.
    """
    try:
        target = resolve_path(path)
        root = get_workspace_root()
    except WorkspaceError as exc:
        return f"error: {exc}"

    if not target.exists() and not target.is_symlink():
        return f"error: path does not exist: {path}"
    if target == root:
        return "error: cannot delete the workspace root"
    try:
        return _delete_target(target, path, recursive=recursive)
    except OSError as exc:
        return f"error: delete failed: {exc}"
