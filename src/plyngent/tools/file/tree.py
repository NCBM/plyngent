from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, get_path_denylist, resolve_path

if TYPE_CHECKING:
    from collections.abc import Sequence
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

# Extra noise dirs skipped by default (in addition to VCS / optional hidden).
# Pass skip_dirs=[] to disable this list (VCS still always skipped).
DEFAULT_NOISE_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        "target",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "coverage",
        ".next",
        ".nuxt",
        ".turbo",
        ".cache",
        "eggs",
        ".eggs",
        "htmlcov",
    }
)


@dataclass(frozen=True, slots=True)
class _TreeLimits:
    max_depth: int
    max_entries: int
    skip_hidden_dirs: bool
    skip_basenames: frozenset[str]
    apply_path_denylist: bool


def _skip_directory(name: str, *, limits: _TreeLimits) -> bool:
    if name in VCS_DIR_NAMES or name in limits.skip_basenames:
        return True
    return bool(limits.skip_hidden_dirs and name.startswith("."))


def _path_denied(path: Path) -> bool:
    """True when resolved path matches a path_denylist substring."""
    denylist = get_path_denylist()
    if not denylist:
        return False
    resolved_str = str(path).replace("\\", "/")
    return any(pattern and pattern.replace("\\", "/") in resolved_str for pattern in denylist)


def _list_children(directory: Path, *, limits: _TreeLimits) -> list[Path] | str:
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
        if is_dir and _skip_directory(child.name, limits=limits):
            continue
        if limits.apply_path_denylist and _path_denied(child):
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

    children = _list_children(directory, limits=limits)
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


def _resolve_skip_basenames(skip_dirs: Sequence[str] | None) -> frozenset[str]:
    """None → default noise set; explicit list (including empty) replaces defaults."""
    if skip_dirs is None:
        return DEFAULT_NOISE_DIR_NAMES
    return frozenset(name for name in skip_dirs if name)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE)
async def tree(
    path: str = ".",
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    skip_hidden_dirs: bool = True,
    skip_dirs: list[str] | None = None,
    apply_path_denylist: bool = True,
) -> str:
    """Show a directory tree under the workspace.

    Always skips VCS metadata directories (``.git``, ``.hg``, ``.svn``, …).
    By default also skips common noise dirs (``node_modules``, ``__pycache__``,
    ``.venv``, ``dist``, …). Pass ``skip_dirs=[]`` to disable the noise list
    (VCS still skipped). Pass an explicit list to replace the default noise set.

    By default skips other dot-directories (not hidden files). Use
    ``skip_hidden_dirs=false`` to include them.

    ``apply_path_denylist`` (default true) hides entries whose full path matches
    the agent ``path_denylist`` policy.

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
    limits = _TreeLimits(
        max_depth=max_depth,
        max_entries=max_entries,
        skip_hidden_dirs=skip_hidden_dirs,
        skip_basenames=_resolve_skip_basenames(skip_dirs),
        apply_path_denylist=apply_path_denylist,
    )
    _render_tree(
        origin,
        prefix="",
        depth=0,
        limits=limits,
        lines=lines,
    )
    return "\n".join(lines)
