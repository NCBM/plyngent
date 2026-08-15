from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from plyngent.agent import ToolTag, tool
from plyngent.tools.workspace import WorkspaceError, get_path_denylist, resolve_path

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Model-facing renderings. ``markdown`` (default) is the cheapest, most
# reliable representation for the agent; ``flat`` gives directly-usable paths;
# ``decorated`` keeps the classic box-drawing tree for human readers.
type TreeFormat = Literal["markdown", "flat", "decorated"]

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


def _render_tree_markdown(
    directory: Path,
    *,
    prefix: str,
    depth: int,
    limits: _TreeLimits,
    lines: list[str],
) -> None:
    """Nested ``- `` bullet list (2-space indent per level)."""
    if depth >= limits.max_depth:
        return

    children = _list_children(directory, limits=limits)
    if isinstance(children, str):
        lines.append(f"{prefix}- {children}")
        return

    truncated = len(children) > limits.max_entries
    shown = children[: limits.max_entries]
    for child in shown:
        try:
            is_dir = child.is_dir()
        except OSError:
            lines.append(f"{prefix}- {child.name} [error: stat failed]")
            continue
        if is_dir:
            lines.append(f"{prefix}- {child.name}/")
            _render_tree_markdown(
                child,
                prefix=prefix + "  ",
                depth=depth + 1,
                limits=limits,
                lines=lines,
            )
        else:
            lines.append(f"{prefix}- {child.name}")

    if truncated:
        more = len(children) - limits.max_entries
        lines.append(f"{prefix}- … ({more} more entries not shown)")


def _flat_paths(directory: Path, *, limits: _TreeLimits) -> list[str]:
    """All paths (dirs with trailing ``/``, files plain) in depth-first order.

    Includes empty directories; bounded by *max_depth* (``max_entries`` is
    applied to the total by the caller).
    """
    out: list[str] = []

    def visit(child_dir: Path, rel: str, depth: int) -> None:
        if depth >= limits.max_depth:
            return
        children = _list_children(child_dir, limits=limits)
        if isinstance(children, str):
            out.append(f"{rel or child_dir.name} [error: cannot list]")
            return
        for child in children:
            try:
                is_dir = child.is_dir()
            except OSError:
                out.append(f"{child.name} [error: stat failed]")
                continue
            child_rel = child.name if not rel else f"{rel}/{child.name}"
            if is_dir:
                out.append(f"{child_rel}/")
                visit(child, child_rel, depth + 1)
            else:
                out.append(child_rel)

    visit(directory, "", 0)
    return out


def _resolve_skip_basenames(skip_dirs: Sequence[str] | None) -> frozenset[str]:
    """None → default noise set; explicit list (including empty) replaces defaults."""
    if skip_dirs is None:
        return DEFAULT_NOISE_DIR_NAMES
    return frozenset(name for name in skip_dirs if name)


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def tree(
    path: str = ".",
    *,
    format: TreeFormat = "markdown",  # noqa: A002 — model-facing param name
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

    ``format`` selects the representation (the model chooses per call):
    ``markdown`` (default) renders a nested ``- `` bullet list — cheapest and
    most reliable for the agent; ``flat`` lists one relative path per line
    (directories with a trailing ``/``) so every path is directly usable
    without reconstructing it; ``decorated`` uses the classic box-drawing tree.

    ``max_depth`` limits how deep directories are expanded (1 = origin + children).
    ``max_entries`` caps entries per directory (``markdown``/``decorated``) or
    the total number of lines (``flat``).
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

    limits = _TreeLimits(
        max_depth=max_depth,
        max_entries=max_entries,
        skip_hidden_dirs=skip_hidden_dirs,
        skip_basenames=_resolve_skip_basenames(skip_dirs),
        apply_path_denylist=apply_path_denylist,
    )

    if format == "flat":
        paths = _flat_paths(origin, limits=limits)
        lines = paths[:max_entries]
        if len(paths) > max_entries:
            lines.append(f"… ({len(paths) - max_entries} more paths not shown)")
        return "\n".join(lines)

    root_label = path.rstrip("/\\") or "."
    lines: list[str] = [] if format == "markdown" else [f"{root_label}/"]
    renderer = _render_tree_markdown if format == "markdown" else _render_tree
    renderer(
        origin,
        prefix="",
        depth=0,
        limits=limits,
        lines=lines,
    )
    return "\n".join(lines)
