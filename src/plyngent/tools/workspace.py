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


class WorkspaceError(ValueError):
    """Raised when a path or command violates workspace policy."""


class _WorkspaceState:
    root: Path | None = None
    path_denylist: tuple[str, ...] = ()
    command_denylist: frozenset[str] = DEFAULT_COMMAND_DENYLIST


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
    """Clear workspace root (mainly for tests)."""
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


def resolve_path(path: str | Path) -> Path:
    """Resolve ``path`` under the workspace root; reject escapes and denylist hits."""
    root = get_workspace_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()
    try:
        _ = resolved.relative_to(root)
    except ValueError as exc:
        msg = f"path escapes workspace root ({root}): {path}"
        raise WorkspaceError(msg) from exc
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
