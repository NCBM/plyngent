from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plyngent.tools.command_scan import unwrap_command

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from plyngent.tools.context import InstanceState, SessionState

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


class AccessMode(IntEnum):
    """Ordered access level for path grants outside the workspace root.

    A grant satisfies a requirement when its value is ``>=`` the requirement
    (``read`` < ``write`` < ``exec``); ``exec`` also implies file writes.
    """

    READ = 1
    WRITE = 2
    EXEC = 3


_ACCESS_MODE_NAMES: dict[str, AccessMode] = {
    "read": AccessMode.READ,
    "write": AccessMode.WRITE,
    "exec": AccessMode.EXEC,
}


def parse_access_mode(value: str) -> AccessMode | None:
    """Parse a case-insensitive mode token (``read`` / ``write`` / ``exec``)."""
    return _ACCESS_MODE_NAMES.get(value.strip().lower())


def mode_covers(granted: AccessMode, required: AccessMode) -> bool:
    """Whether a grant at *granted* satisfies a tool that needs *required*."""
    return granted >= required


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
    # Static TOML pre-allow (resolved path → mode); never prompts.
    config_allow: dict[Path, AccessMode] = field(default_factory=dict)
    # Process-scoped YOLO / --yes grants (resolved path → mode); never persisted.
    yolo_allow: dict[Path, AccessMode] = field(default_factory=dict)
    # Host-contributed read-only roots (skill directories): every path tool may
    # read them, nothing may write them. Not a grant, so they stay out of
    # ``/grants``; writes go through the owning tool's own confirm.
    static_read: list[Path] = field(default_factory=list)


def _bound_instance() -> InstanceState | None:
    from plyngent.tools.context import get_instance

    return get_instance()


def _bound_session() -> SessionState | None:
    from plyngent.tools.context import get_session

    return get_session()


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


def granted_access_mode(resolved: Path) -> AccessMode | None:
    """Highest config / session / process / static-read mode covering *resolved*."""
    policy = active_workspace_policy()
    session = _bound_session()
    buckets: list[Mapping[Path, AccessMode]] = [policy.config_allow, policy.yolo_allow]
    if session is not None:
        buckets.append(session.access_grants)
    best: AccessMode | None = None
    for grants in buckets:
        for root, granted in grants.items():
            if (resolved == root or resolved.is_relative_to(root)) and (best is None or granted > best):
                best = granted
    for root in policy.static_read:
        if (resolved == root or resolved.is_relative_to(root)) and (best is None or best < AccessMode.READ):
            best = AccessMode.READ
    return best


def set_static_read_roots(
    roots: Sequence[Path | str],
    *,
    instance: InstanceState | None = None,
) -> list[Path]:
    """Install host-contributed read-only roots (resolved); returns them.

    Replaces any previous list. Roots that are not existing directories are
    dropped: an absent skill directory holds nothing to read, and a missing
    entry must never turn into an implicit write permission later.
    """
    inst = instance if instance is not None else require_bound_instance()
    resolved: list[Path] = []
    for root in roots:
        try:
            path = Path(root).expanduser().resolve()
        except OSError:
            continue
        if path.is_dir():
            resolved.append(path)
    inst.workspace.static_read[:] = resolved
    return resolved


def _path_grant_covers(resolved: Path, required: AccessMode) -> bool:
    """Whether a config / session / process grant covers *resolved*."""
    granted = granted_access_mode(resolved)
    return granted is not None and mode_covers(granted, required)


def within_workspace_roots(resolved: Path) -> bool:
    """Whether *resolved* is under the primary root(s) or a temp allowlist root."""
    instance = require_bound_instance()
    return _under_any_root(resolved, instance, instance.workspace)


def denylist_match(resolved: Path, *, directory: bool = False) -> str | None:
    """Return the path denylist pattern matching *resolved*, or ``None``.

    *directory* also matches a trailing-separator pattern (``/secret/``) against
    the directory path itself, not just its children.
    """
    resolved_str = str(resolved).replace("\\", "/")
    if directory and not resolved_str.endswith("/"):
        resolved_str += "/"
    for pattern in active_workspace_policy().path_denylist:
        if pattern and pattern.replace("\\", "/") in resolved_str:
            return pattern
    return None


def resolve_path(path: str | Path, *, required: AccessMode = AccessMode.WRITE) -> Path:
    """Resolve ``path`` under the workspace root, an allowlist root, or a grant.

    Relative paths resolve against the **primary** workspace root. Absolute
    paths may also land under a temporary workspace allowlist entry or a
    directory-access grant; ``required`` is the access mode the caller needs
    (default :attr:`AccessMode.WRITE`, the conservative choice).
    """
    instance = require_bound_instance()
    policy = instance.workspace
    root = get_workspace_root()
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.expanduser().resolve()
    if not _under_any_root(resolved, instance, policy) and not _path_grant_covers(resolved, required):
        msg = (
            f"path escapes workspace root ({root}): {path}; "
            "call request_directory_access to request access to a directory outside the workspace"
        )
        raise WorkspaceError(msg)
    # Normalize separators so denylist entries like ``/secrets/`` match on Windows.
    matched = denylist_match(resolved)
    if matched is not None:
        msg = f"path denied by policy (matched {matched!r}): {path}"
        raise WorkspaceError(msg)
    return resolved


def argv_shape_error(argv: object) -> str | None:
    """Return a model-facing error when ``argv`` is not a non-empty argv list.

    Models occasionally pass a shell-style string or a JSON-encoded array as a
    string; duck-typing would otherwise let it reach ``create_subprocess_exec``
    and splat into single characters. Declared as ``object`` so the isinstance
    checks are not flagged as unnecessary.
    """
    if isinstance(argv, str):
        return 'only accepts an array of args, not a string; pass e.g. ["ls", "-la"]'
    if not isinstance(argv, list):
        return f"only accepts an array of args, not a {type(argv).__name__}"
    if not all(isinstance(part, str) for part in cast("list[object]", argv)):
        return "only accepts an array of string args"
    return None


def check_command_allowed(argv: list[str]) -> None:
    """Raise if argv is empty or a basename in the command chain is denylisted.

    The chain is the program that really runs plus every wrapper it is reached
    through (``env FOO=1 rm``, ``pdm run rm``, ``sudo -u x rm``), so a denylisted
    program cannot hide behind one. Basenames are compared lower-cased, so
    ``RM`` and ``rm.exe`` count as ``rm``.

    Denylisted basenames are not hard-rejected when a policy confirm hook is
    installed: the human is asked (with a timeout; default deny). Session grants
    skip re-prompting for the same basename. Independent of YOLO soft-confirm.
    """
    shape_error = argv_shape_error(argv)
    if shape_error is not None:
        msg = f"command {shape_error}"
        raise WorkspaceError(msg)
    if not argv:
        msg = "command argv must not be empty"
        raise WorkspaceError(msg)
    policy = active_workspace_policy()
    scan = unwrap_command(argv)
    for binary in (*scan.wrappers, scan.base):
        if binary not in policy.command_denylist or binary in policy.policy_allowed_commands:
            continue
        hook = policy.policy_confirm_hook
        if hook is None:
            msg = f"command denied by policy (basename {binary!r} is blocked)"
            raise WorkspaceError(msg)
        timeout = policy.policy_confirm_timeout_seconds
        try:
            allowed = bool(hook(binary, list(argv), timeout))
        except Exception as exc:
            msg = f"command denied by policy (basename {binary!r}; confirm failed: {exc})"
            raise WorkspaceError(msg) from exc
        if not allowed:
            msg = (
                f"command denied by policy (basename {binary!r} is blocked; "
                f"user declined or timed out after {timeout:g}s)"
            )
            raise WorkspaceError(msg)
        policy.policy_allowed_commands.add(binary)
