"""Directory-access grants: request tool plus config / session / process stores.

A grant is one ``path → mode`` pair; its *source* is structural:

- :attr:`WorkspacePolicy.config_allow` — static TOML pre-allow (never prompts),
- :attr:`SessionState.access_grants` — human-approved for this session,
- :attr:`WorkspacePolicy.yolo_allow` — ``--yes`` / YOLO auto-approval, process only.

``resolve_path`` consults all three. Grants are a policy gate for path-resolving
tools, **not** a sandbox: an ``exec`` grant (or any command) can reach the rest
of the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from plyngent.agent import ToolTag, tool
from plyngent.tools.context import get_session, require_session
from plyngent.tools.workspace import (
    AccessMode,
    denylist_match,
    get_policy_confirm_timeout,
    granted_access_mode,
    mode_covers,
    parse_access_mode,
    require_bound_instance,
    within_workspace_roots,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from plyngent.tools.context import InstanceState, SessionState

# Max active grants across all three stores (bounds prompt/listing noise).
MAX_ACCESS_GRANTS = 32

_HOOK_KEY = "directory_access_confirm_hook"


class AccessDecision(NamedTuple):
    """Hook approval for one directory-access request.

    ``mode`` is the granted level (may differ from the requested one);
    ``persist`` is true for a session grant, false for a process-only grant.
    """

    mode: AccessMode
    persist: bool = True


type DirectoryAccessConfirmHook = Callable[[Path, AccessMode, str, float], AccessDecision | str | None]


def get_directory_access_confirm_hook(instance: InstanceState | None = None) -> DirectoryAccessConfirmHook | None:
    """Return the host-installed confirm hook for this instance, if any."""
    inst = instance if instance is not None else require_bound_instance()
    hook = inst.extras.get(_HOOK_KEY)
    if hook is None or not callable(hook):
        return None

    def _as_hook(path: Path, mode: AccessMode, reason: str, timeout: float) -> AccessDecision | str | None:
        result: object = hook(path, mode, reason, timeout)
        if isinstance(result, AccessDecision | str):
            return result
        return None

    return _as_hook


def set_directory_access_confirm_hook(
    hook: DirectoryAccessConfirmHook | None,
    *,
    instance: InstanceState | None = None,
) -> None:
    """Install (or clear) the confirm hook the CLI uses to ask the human."""
    inst = instance if instance is not None else require_bound_instance()
    inst.extras[_HOOK_KEY] = hook


def grant_session_access(
    path: Path | str,
    mode: AccessMode,
    *,
    session: SessionState | None = None,
) -> Path:
    """Grant *mode* on *path* for the bound session; returns the resolved path."""
    resolved = Path(path).expanduser().resolve()
    sess = session if session is not None else require_session()
    sess.access_grants[resolved] = mode
    return resolved


def grant_process_access(path: Path | str, mode: AccessMode) -> Path:
    """Grant *mode* on *path* for this process only (``--yes``); never persisted."""
    resolved = Path(path).expanduser().resolve()
    require_bound_instance().workspace.yolo_allow[resolved] = mode
    return resolved


def clear_process_access() -> None:
    """Drop process-scoped (``--yes``) grants."""
    require_bound_instance().workspace.yolo_allow.clear()


def clear_session_access(*, session: SessionState | None = None) -> None:
    """Drop live session grants (durable copy is cleared by the host)."""
    sess = session if session is not None else get_session()
    if sess is not None:
        sess.access_grants.clear()


def active_grant_count() -> int:
    """Total live grants across config, session, and process stores."""
    policy = require_bound_instance().workspace
    total = len(policy.config_allow) + len(policy.yolo_allow)
    session = get_session()
    if session is not None:
        total += len(session.access_grants)
    return total


def set_config_access(entries: Mapping[str, str]) -> tuple[list[Path], list[str]]:
    """Install static TOML pre-allow entries on the bound instance policy.

    Replaces any previous config grants. Returns ``(applied roots, skipped
    keys)``; entries are skipped when the mode is unknown or the path does not
    exist (the host warns about skipped keys).
    """
    policy = require_bound_instance().workspace
    policy.config_allow.clear()
    applied: list[Path] = []
    skipped: list[str] = []
    for raw_path, raw_mode in entries.items():
        mode = parse_access_mode(raw_mode)
        resolved: Path | None
        try:
            resolved = Path(raw_path).expanduser().resolve()
        except OSError:
            resolved = None
        if mode is None or resolved is None or not resolved.exists():
            skipped.append(raw_path)
            continue
        policy.config_allow[resolved] = mode
        applied.append(resolved)
    return applied, skipped


def _resolve_target(path: str) -> Path | str:
    """Resolve a requested path; return a model-facing error string on failure."""
    try:
        target = Path(path).expanduser().resolve()
    except OSError as exc:
        return f"error: cannot resolve path {path!r}: {exc}"
    if not target.exists():
        return f"error: path does not exist: {target}"
    return target


def _precheck_target(target: Path, requested: AccessMode) -> str | None:
    """Return a short-circuit message when no prompt is needed, else ``None``."""
    if requested is AccessMode.EXEC and not target.is_dir():
        return f"error: exec access requires a directory: {target}"
    matched = denylist_match(target, directory=target.is_dir())
    if matched is not None:
        return f"error: path denied by policy (matched {matched!r}); it can never be granted: {target}"
    if within_workspace_roots(target):
        return f"already accessible: {target} is inside the workspace"
    granted = granted_access_mode(target)
    if granted is not None and mode_covers(granted, requested):
        return f"already granted: {target} ({granted.name.lower()})"
    if active_grant_count() >= MAX_ACCESS_GRANTS:
        return f"error: too many active access grants (max {MAX_ACCESS_GRANTS}); revoke some first"
    return None


def _apply_decision(target: Path, decision: AccessDecision | str | None, timeout: float) -> str:
    """Record an approved grant or format the denial for the model."""
    if isinstance(decision, AccessDecision):
        if decision.persist and get_session() is not None:
            _ = grant_session_access(target, decision.mode)
            scope = "session"
        else:
            _ = grant_process_access(target, decision.mode)
            scope = "process"
        return (
            f"access granted: {target} ({decision.mode.name.lower()}, {scope})\n"
            "note: the path denylist still applies; the human can revoke with /grants"
        )
    if isinstance(decision, str) and decision.strip():
        return f"error: access to {target} denied by user; user comment: {decision.strip()}"
    return f"error: access to {target} denied (declined or timed out after {timeout:g}s)"


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.SESSION_STATE)
async def request_directory_access(path: str, reason: str = "", mode: str = "read") -> str:
    """Request human approval to use a directory (or exact file) outside the workspace.

    Call this instead of retrying a path that failed with ``path escapes
    workspace root``. ``mode`` is ``read`` (inspect files), ``write`` (also
    modify files) or ``exec`` (also use it as ``run_argv`` / PTY cwd); the human
    may approve a different level. Approval lasts for the session (or for this
    process when ``--yes`` auto-approves). Paths matching the configured
    denylist can never be granted.
    """
    requested = parse_access_mode(mode)
    if requested is None:
        return f"error: invalid mode {mode!r} (use read, write, or exec)"
    resolved = _resolve_target(path)
    if isinstance(resolved, str):
        return resolved
    target = resolved
    precheck = _precheck_target(target, requested)
    if precheck is not None:
        return precheck
    hook = get_directory_access_confirm_hook()
    if hook is None:
        return f"error: access to {target} requires a human confirm (no confirm hook installed; denied)"
    timeout = get_policy_confirm_timeout()
    try:
        decision = hook(target, requested, reason, timeout)
    except Exception as exc:  # noqa: BLE001 — surface confirm failures to the model
        return f"error: access confirm for {target} failed: {exc}"
    return _apply_decision(target, decision, timeout)


ACCESS_TOOLS = [
    request_directory_access,
]
