from __future__ import annotations

from pathlib import Path

DEFAULT_COMMAND_DENYLIST: frozenset[str] = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "pkexec",
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "useradd",
        "userdel",
        "passwd",
        "chmod",
        "chown",
        "mount",
        "umount",
    }
)

# Max concurrent temporary workspaces registered in one process.
MAX_TEMPORARY_WORKSPACES = 16


class WorkspaceError(ValueError):
    """Raised when a path or command violates workspace policy."""


class _WorkspaceState:
    root: Path | None = None
    path_denylist: tuple[str, ...] = ()
    command_denylist: frozenset[str] = DEFAULT_COMMAND_DENYLIST
    # Extra roots allowed for resolve_path (e.g. temporary workspaces under system temp).
    allowlist: list[Path]
    # Paths created by new_temporary_workspace (subset of allowlist); cleaned on chat exit.
    temporary_owned: list[Path]

    def __init__(self) -> None:
        self.allowlist = []
        self.temporary_owned = []


_state = _WorkspaceState()


def set_workspace_root(root: Path | str) -> Path:
    """Set the workspace root used by tools; returns the resolved root."""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        msg = f"workspace root is not a directory: {path}"
        raise WorkspaceError(msg)
    _state.root = path
    return path


def get_workspace_root() -> Path:
    """Return the configured workspace root."""
    if _state.root is None:
        msg = "workspace root is not set; call set_workspace_root() first"
        raise WorkspaceError(msg)
    return _state.root


def clear_workspace_root() -> None:
    """Clear workspace root (mainly for tests). Does not clear allowlist."""
    _state.root = None


def set_path_denylist(patterns: list[str] | tuple[str, ...] | None) -> None:
    """Set path substring denylist (matched against resolved path strings)."""
    _state.path_denylist = tuple(patterns or ())


def get_path_denylist() -> tuple[str, ...]:
    """Return the current path substring denylist."""
    return _state.path_denylist


def set_command_denylist(names: list[str] | tuple[str, ...] | frozenset[str] | None) -> None:
    """Set denied command basenames (None restores defaults)."""
    _state.command_denylist = DEFAULT_COMMAND_DENYLIST if names is None else frozenset(names)


def get_command_denylist() -> frozenset[str]:
    return _state.command_denylist


def add_workspace_allowlist(root: Path | str, *, owned: bool = False) -> Path:
    """Allow tool paths under *root* in addition to the primary workspace.

    When *owned* is true, the path is also registered for chat-exit cleanup
    (only paths created by :func:`new_temporary_workspace`).
    """
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        msg = f"allowlist root is not a directory: {path}"
        raise WorkspaceError(msg)
    if path not in _state.allowlist:
        if len(_state.allowlist) >= MAX_TEMPORARY_WORKSPACES and owned:
            msg = f"too many temporary workspaces (max {MAX_TEMPORARY_WORKSPACES})"
            raise WorkspaceError(msg)
        _state.allowlist.append(path)
    if owned and path not in _state.temporary_owned:
        _state.temporary_owned.append(path)
    return path


def list_workspace_allowlist() -> list[Path]:
    """Return a copy of extra allowed roots (not including the primary workspace)."""
    return list(_state.allowlist)


def clear_workspace_allowlist() -> None:
    """Clear allowlist and owned-temp registry (tests). Does not delete directories."""
    _state.allowlist.clear()
    _state.temporary_owned.clear()


def pop_owned_temporary_workspaces() -> list[Path]:
    """Return and clear the owned temporary workspace list (for chat-exit cleanup).

    Paths remain on the allowlist until the caller removes them via
    :func:`remove_workspace_allowlist`.
    """
    owned = list(_state.temporary_owned)
    _state.temporary_owned.clear()
    return owned


def remove_workspace_allowlist(root: Path | str) -> None:
    """Drop *root* from the allowlist if present."""
    path = Path(root).expanduser().resolve()
    while path in _state.allowlist:
        _state.allowlist.remove(path)


def _under_any_root(resolved: Path) -> bool:
    roots: list[Path] = []
    if _state.root is not None:
        roots.append(_state.root)
    roots.extend(_state.allowlist)
    for root in roots:
        try:
            _ = resolved.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` under the workspace root or an allowlisted temp root.

    Relative paths resolve against the **primary** workspace root. Absolute
    paths may also land under a temporary workspace allowlist entry.
    """
    root = get_workspace_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()
    if not _under_any_root(resolved):
        msg = f"path escapes workspace root ({root}): {path}"
        raise WorkspaceError(msg)
    # Normalize separators so denylist entries like ``/secrets/`` match on Windows.
    resolved_str = str(resolved).replace("\\", "/")
    for pattern in _state.path_denylist:
        if pattern and pattern.replace("\\", "/") in resolved_str:
            msg = f"path denied by policy (matched {pattern!r}): {path}"
            raise WorkspaceError(msg)
    return resolved


def check_command_allowed(argv: list[str]) -> None:
    """Raise if argv is empty or the executable basename is denylisted."""
    if not argv:
        msg = "command argv must not be empty"
        raise WorkspaceError(msg)
    binary = Path(argv[0]).name
    if binary in _state.command_denylist:
        msg = f"command denied by policy (basename {binary!r} is blocked)"
        raise WorkspaceError(msg)
