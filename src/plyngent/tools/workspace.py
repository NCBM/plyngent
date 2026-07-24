from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.tools.context import InstanceState

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

# Timed human override for command denylist (independent of YOLO soft-confirm).
DEFAULT_POLICY_CONFIRM_TIMEOUT_SECONDS = 30.0

# Hook: (basename, argv, timeout_seconds) -> True allow for this session basename.
type PolicyConfirmHook = Callable[[str, Sequence[str], float], bool]


class WorkspaceError(ValueError):
    """Raised when a path or command violates workspace policy."""


@dataclass
class WorkspacePolicy:
    """Workspace path/command policy for one agent host / instance.

    Lives on :class:`~plyngent.tools.context.InstanceState.workspace`. There is
    no process-global policy bag — hosts must bind instance state around tool use.
    """

    root: Path | None = None
    path_denylist: tuple[str, ...] = ()
    command_denylist: frozenset[str] = DEFAULT_COMMAND_DENYLIST
    allowlist: list[Path] = field(default_factory=list)
    temporary_owned: list[Path] = field(default_factory=list)
    policy_allowed_commands: set[str] = field(default_factory=set)
    policy_confirm_hook: PolicyConfirmHook | None = None
    policy_confirm_timeout_seconds: float = DEFAULT_POLICY_CONFIRM_TIMEOUT_SECONDS


def _bound_instance() -> InstanceState | None:
    from plyngent.tools.context import get_instance

    return get_instance()


def require_bound_instance() -> InstanceState:
    """Return the bound instance or raise :class:`WorkspaceError`."""
    instance = _bound_instance()
    if instance is None:
        msg = "instance state is not bound; host must set InstanceState around tool execution"
        raise WorkspaceError(msg)
    return instance


def active_workspace_policy() -> WorkspacePolicy:
    """Return the policy bag for the bound instance (required)."""
    return require_bound_instance().workspace


def set_workspace_root(root: Path | str) -> Path:
    """Set the workspace root on the bound instance; returns the resolved root."""
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        msg = f"workspace root is not a directory: {path}"
        raise WorkspaceError(msg)
    instance = require_bound_instance()
    instance.workspace.root = path
    instance.workspace_root = path
    return path


def get_workspace_root() -> Path:
    """Return the bound instance workspace root."""
    instance = require_bound_instance()
    if instance.workspace_root is not None:
        return instance.workspace_root
    if instance.workspace.root is not None:
        return instance.workspace.root
    msg = "workspace root is not set on the bound instance"
    raise WorkspaceError(msg)


def clear_workspace_root() -> None:
    """Clear workspace root on the bound instance (mainly for tests)."""
    instance = require_bound_instance()
    instance.workspace.root = None
    instance.workspace_root = None


def set_path_denylist(patterns: list[str] | tuple[str, ...] | None) -> None:
    """Set path substring denylist (matched against resolved path strings)."""
    active_workspace_policy().path_denylist = tuple(patterns or ())


def get_path_denylist() -> tuple[str, ...]:
    """Return the current path substring denylist."""
    return active_workspace_policy().path_denylist


def set_command_denylist(names: list[str] | tuple[str, ...] | frozenset[str] | None) -> None:
    """Set denied command basenames (None restores defaults)."""
    policy = active_workspace_policy()
    policy.command_denylist = DEFAULT_COMMAND_DENYLIST if names is None else frozenset(names)
    policy.policy_allowed_commands &= policy.command_denylist


def get_command_denylist() -> frozenset[str]:
    return active_workspace_policy().command_denylist


def set_policy_confirm_hook(hook: PolicyConfirmHook | None) -> None:
    """Register a timed human confirm for denylisted commands (CLI installs this)."""
    active_workspace_policy().policy_confirm_hook = hook


def get_policy_confirm_hook() -> PolicyConfirmHook | None:
    return active_workspace_policy().policy_confirm_hook


def set_policy_confirm_timeout(seconds: float) -> None:
    """Timeout for policy confirm prompts (must be > 0)."""
    if seconds <= 0:
        msg = "policy confirm timeout must be > 0"
        raise WorkspaceError(msg)
    active_workspace_policy().policy_confirm_timeout_seconds = float(seconds)


def get_policy_confirm_timeout() -> float:
    return active_workspace_policy().policy_confirm_timeout_seconds


def clear_policy_allowed_commands() -> None:
    """Drop session-scoped denylist overrides (tests / chat exit)."""
    active_workspace_policy().policy_allowed_commands.clear()


def grant_policy_command(basename: str) -> None:
    """Allow *basename* for this instance despite the denylist."""
    name = basename.strip()
    if name:
        active_workspace_policy().policy_allowed_commands.add(name)


def add_workspace_allowlist(root: Path | str, *, owned: bool = False) -> Path:
    """Allow tool paths under *root* in addition to the primary workspace.

    When *owned* is true, the path is also registered for chat-exit cleanup
    (only paths created by :func:`new_temporary_workspace`).
    """
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        msg = f"allowlist root is not a directory: {path}"
        raise WorkspaceError(msg)
    policy = active_workspace_policy()
    if path not in policy.allowlist:
        if len(policy.allowlist) >= MAX_TEMPORARY_WORKSPACES and owned:
            msg = f"too many temporary workspaces (max {MAX_TEMPORARY_WORKSPACES})"
            raise WorkspaceError(msg)
        policy.allowlist.append(path)
    if owned and path not in policy.temporary_owned:
        policy.temporary_owned.append(path)
    return path


def list_workspace_allowlist() -> list[Path]:
    """Return a copy of extra allowed roots (not including the primary workspace)."""
    return list(active_workspace_policy().allowlist)


def clear_workspace_allowlist() -> None:
    """Clear allowlist and owned-temp registry (tests). Does not delete directories."""
    policy = active_workspace_policy()
    policy.allowlist.clear()
    policy.temporary_owned.clear()


def pop_owned_temporary_workspaces() -> list[Path]:
    """Return and clear the owned temporary workspace list (for chat-exit cleanup).

    Paths remain on the allowlist until the caller removes them via
    :func:`remove_workspace_allowlist`.
    """
    policy = active_workspace_policy()
    owned = list(policy.temporary_owned)
    policy.temporary_owned.clear()
    return owned


def remove_workspace_allowlist(root: Path | str) -> None:
    """Drop *root* from the allowlist if present."""
    path = Path(root).expanduser().resolve()
    policy = active_workspace_policy()
    while path in policy.allowlist:
        policy.allowlist.remove(path)


def _primary_roots(instance: InstanceState, policy: WorkspacePolicy) -> list[Path]:
    roots: list[Path] = []
    if instance.workspace_root is not None:
        roots.append(instance.workspace_root)
    if policy.root is not None and policy.root not in roots:
        roots.append(policy.root)
    return roots


def _under_any_root(resolved: Path, instance: InstanceState, policy: WorkspacePolicy) -> bool:
    roots = _primary_roots(instance, policy)
    roots.extend(policy.allowlist)
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
    instance = require_bound_instance()
    policy = instance.workspace
    root = get_workspace_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()
    if not _under_any_root(resolved, instance, policy):
        msg = f"path escapes workspace root ({root}): {path}"
        raise WorkspaceError(msg)
    # Normalize separators so denylist entries like ``/secrets/`` match on Windows.
    resolved_str = str(resolved).replace("\\", "/")
    for pattern in policy.path_denylist:
        if pattern and pattern.replace("\\", "/") in resolved_str:
            msg = f"path denied by policy (matched {pattern!r}): {path}"
            raise WorkspaceError(msg)
    return resolved


def check_command_allowed(argv: list[str]) -> None:
    """Raise if argv is empty or the executable basename is denylisted.

    Denylisted basenames are not hard-rejected when a policy confirm hook is
    installed: the human is asked (with a timeout; default deny). Session grants
    skip re-prompting for the same basename. Independent of YOLO soft-confirm.
    """
    if not argv:
        msg = "command argv must not be empty"
        raise WorkspaceError(msg)
    policy = active_workspace_policy()
    binary = Path(argv[0]).name
    if binary not in policy.command_denylist:
        return
    if binary in policy.policy_allowed_commands:
        return
    hook = policy.policy_confirm_hook
    if hook is not None:
        timeout = policy.policy_confirm_timeout_seconds
        try:
            allowed = bool(hook(binary, list(argv), timeout))
        except Exception as exc:
            msg = f"command denied by policy (basename {binary!r}; confirm failed: {exc})"
            raise WorkspaceError(msg) from exc
        if allowed:
            policy.policy_allowed_commands.add(binary)
            return
        msg = (
            f"command denied by policy (basename {binary!r} is blocked; user declined or timed out after {timeout:g}s)"
        )
        raise WorkspaceError(msg)
    msg = f"command denied by policy (basename {binary!r} is blocked)"
    raise WorkspaceError(msg)
