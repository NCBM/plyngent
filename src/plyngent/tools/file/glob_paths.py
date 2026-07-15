from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import tool
from plyngent.tools.file.tree import VCS_DIR_NAMES
from plyngent.tools.workspace import WorkspaceError, get_workspace_root, resolve_path

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MAX_MATCHES = 200


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _hidden_or_vcs(parts: tuple[str, ...]) -> bool:
    return any(part in VCS_DIR_NAMES or part.startswith(".") for part in parts)


def _collect_glob(
    base: Path,
    root: Path,
    pattern: str,
    *,
    max_matches: int,
    skip_hidden_dirs: bool,
) -> tuple[list[str], bool] | str:
    matches: list[str] = []
    try:
        candidates = sorted(base.glob(pattern), key=lambda p: str(p).casefold())
    except (OSError, ValueError) as exc:
        return f"error: glob failed: {exc}"

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            rel = resolved.relative_to(root)
        except OSError, ValueError:
            continue
        # Skip anything under (or itself) VCS / hidden path components.
        if skip_hidden_dirs and _hidden_or_vcs(rel.parts):
            continue
        if not resolved.is_file() and not resolved.is_dir():
            continue
        matches.append(_rel(resolved, root))
        if len(matches) >= max_matches:
            return matches, True
    return matches, False


def _resolve_glob_base(path: str) -> tuple[Path, Path] | str:
    try:
        root = get_workspace_root()
        base = resolve_path(path)
    except WorkspaceError as exc:
        return f"error: {exc}"
    if not base.is_dir():
        return f"error: not a directory: {path}"
    return root, base


@tool
def glob_paths(
    pattern: str,
    path: str = ".",
    *,
    max_matches: int = DEFAULT_MAX_MATCHES,
    skip_hidden_dirs: bool = True,
) -> str:
    """Find files under the workspace matching a glob pattern (e.g. ``**/*.py``).

    Search is relative to ``path`` (default workspace root). Skips VCS dirs
    (``.git``, …) and hidden directories by default. Returns paths relative to
    the workspace root, one per line.
    """
    if not pattern.strip() or max_matches < 1:
        return "error: pattern must not be empty and max_matches must be >= 1"

    resolved = _resolve_glob_base(path)
    if isinstance(resolved, str):
        return resolved
    root, base = resolved

    result = _collect_glob(base, root, pattern, max_matches=max_matches, skip_hidden_dirs=skip_hidden_dirs)
    if isinstance(result, str):
        return result
    matches, truncated = result
    if not matches:
        return "(no matches)"
    body = "\n".join(matches)
    if truncated:
        body += f"\n...[truncated at {max_matches} matches]"
    return body
