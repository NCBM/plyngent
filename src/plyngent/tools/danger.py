from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.tools.workspace import WorkspaceError, resolve_path

if TYPE_CHECKING:
    from collections.abc import Mapping


def _write_file_reason(args: Mapping[str, object]) -> str | None:
    path = args.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        target = resolve_path(path)
    except WorkspaceError:
        return None
    if target.exists() or target.is_symlink():
        return f"overwrite existing file {path!r}"
    return None


def classify_danger(name: str, args: Mapping[str, object]) -> str | None:
    """Return a short reason if ``name``/``args`` need user confirm, else ``None``.

    Hard denylists (paths/commands) still raise independently. This only covers
    soft confirms for mutating tools that policy otherwise allows.
    """
    reasons: dict[str, str | None] = {
        "delete_path": (
            f"delete path {args.get('path', '')!r} recursively"
            if bool(args.get("recursive", False))
            else f"delete path {args.get('path', '')!r}"
        ),
        "move_path": f"move {args.get('src', '')!r} -> {args.get('dst', '')!r}",
        "copy_path": (
            f"copy with overwrite {args.get('src', '')!r} -> {args.get('dst', '')!r}"
            if bool(args.get("overwrite", False))
            else None
        ),
        "write_file": _write_file_reason(args),
    }
    if name not in reasons:
        return None
    return reasons[name]
