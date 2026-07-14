from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.agent import tool
from plyngent.tools.workspace import WorkspaceError, resolve_path

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_ENTRIES = 50

# Always skipped directory basenames (VCS metadata).
VCS_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".bzr",
        "CVS",
        ".jj",
        "_darcs",
        ".fossil",
    }
)


@dataclass(frozen=True, slots=True)
class _TreeLimits:
    max_depth: int
    max_entries: int
    skip_hidden_dirs: bool


def _skip_directory(name: str, *, skip_hidden_dirs: bool) -> bool:
    if name in VCS_DIR_NAMES:
        return True
    return bool(skip_hidden_dirs and name.startswith("."))


def _list_children(directory: Path, *, skip_hidden_dirs: bool) -> list[Path] | str:
    try:
        children = list(directory.iterdir())
    except OSError as exc:
        return f"error: cannot list {directory.name}: {exc}"
    visible: list[Path] = []
    for child in children:
        try:
            is_dir = child.is_dir()
        except OSError:
            continue
        if is_dir and _skip_directory(child.name, skip_hidden_dirs=skip_hidden_dirs):
            continue
        visible.append(child)
    # Directories first, then files; alphabetical within each group.
    visible.sort(key=lambda p: (not p.is_dir(), p.name.casefold()))
    return visible


def _render_tree(
    directory: Path,
    *,
    prefix: str,
    depth: int,
    limits: _TreeLimits,
    lines: list[str],
) -> None:
    if depth >= limits.max_depth:
        return

    children = _list_children(directory, skip_hidden_dirs=limits.skip_hidden_dirs)
    if isinstance(children, str):
        lines.append(f"{prefix}{children}")
        return

    truncated = len(children) > limits.max_entries
    shown = children[: limits.max_entries]
    for index, child in enumerate(shown):
        is_last = index == len(shown) - 1 and not truncated
        branch = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        try:
            is_dir = child.is_dir()
        except OSError:
            lines.append(f"{prefix}{branch}{child.name} [error: stat failed]")
            continue
        if is_dir:
            lines.append(f"{prefix}{branch}{child.name}/")
            if depth + 1 < limits.max_depth:
                _render_tree(
                    child,
                    prefix=prefix + extension,
                    depth=depth + 1,
                    limits=limits,
                    lines=lines,
                )
        else:
            lines.append(f"{prefix}{branch}{child.name}")

    if truncated:
        more = len(children) - limits.max_entries
        lines.append(f"{prefix}└── … ({more} more entries not shown)")


@tool
def tree(
    path: str = ".",
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    skip_hidden_dirs: bool = True,
) -> str:
    """Show a directory tree under the workspace.

    Always skips VCS metadata directories (``.git``, ``.hg``, ``.svn``, …).
    By default skips other dot-directories (not hidden files). Use
    ``skip_hidden_dirs=false`` to include them.

    ``max_depth`` limits how deep directories are expanded (1 = origin + children).
    ``max_entries`` caps how many entries are listed per directory.
    """
    if max_depth < 1:
        return "error: max_depth must be >= 1"
    if max_entries < 1:
        return "error: max_entries must be >= 1"

    try:
        origin = resolve_path(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not origin.is_dir():
        return f"error: not a directory: {path}"

    root_label = path.rstrip("/\\") or "."
    lines = [f"{root_label}/"]
    _render_tree(
        origin,
        prefix="",
        depth=0,
        limits=_TreeLimits(
            max_depth=max_depth,
            max_entries=max_entries,
            skip_hidden_dirs=skip_hidden_dirs,
        ),
        lines=lines,
    )
    return "\n".join(lines)
