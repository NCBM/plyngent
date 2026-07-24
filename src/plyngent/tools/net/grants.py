"""Instance-scoped private-fetch host grants (not YOLO-skippable)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.tools.net.policy import FetchPolicyError, HostClass, assess_host, grant_key, parse_fetch_url
from plyngent.tools.workspace import WorkspaceError, require_bound_instance

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from plyngent.tools.context import InstanceState

# (host, port, url, timeout_seconds) -> True to allow for this process/instance.
type FetchPolicyConfirmHook = Callable[[str, int, str, float], bool]


def get_private_grants(instance: InstanceState | None = None) -> set[str]:
    """Return the live grant key set (host:port strings)."""
    inst = instance if instance is not None else require_bound_instance()
    bag_obj = inst.extras.get("fetch_private_grants")
    if isinstance(bag_obj, set):
        return cast("set[str]", bag_obj)
    empty: set[str] = set()
    inst.extras["fetch_private_grants"] = empty
    return empty


def grant_private_host(host: str, port: int, *, instance: InstanceState | None = None) -> str:
    """Record a private host:port grant; returns the grant key."""
    key = grant_key(host, port)
    get_private_grants(instance).add(key)
    return key


def clear_private_grants(*, instance: InstanceState | None = None) -> None:
    get_private_grants(instance).clear()


def has_private_grant(host: str, port: int, *, instance: InstanceState | None = None) -> bool:
    return grant_key(host, port) in get_private_grants(instance)


def get_fetch_policy_confirm_hook(instance: InstanceState | None = None) -> FetchPolicyConfirmHook | None:
    inst = instance if instance is not None else require_bound_instance()
    hook = inst.extras.get("fetch_policy_confirm_hook")
    if hook is None or not callable(hook):
        return None

    def _as_hook(host: str, port: int, url: str, timeout_seconds: float) -> bool:
        return bool(hook(host, port, url, timeout_seconds))

    return _as_hook


def set_fetch_policy_confirm_hook(
    hook: FetchPolicyConfirmHook | None,
    *,
    instance: InstanceState | None = None,
) -> None:
    inst = instance if instance is not None else require_bound_instance()
    inst.extras["fetch_policy_confirm_hook"] = hook


async def ensure_host_allowed(
    url: str,
    *,
    policy_timeout_seconds: float | None = None,
    instance: InstanceState | None = None,
) -> None:
    """Raise :class:`FetchPolicyError` if *url*'s host may not be contacted.

    Public hosts pass. Forbidden (metadata) always fail. Private/loopback
    requires an instance grant or a successful policy confirm hook (never YOLO).
    """
    try:
        inst = instance if instance is not None else require_bound_instance()
    except WorkspaceError as exc:
        msg = f"instance state is not bound for fetch policy: {exc}"
        raise FetchPolicyError(msg) from exc

    parsed = parse_fetch_url(url)
    assessment = await assess_host(parsed.host, parsed.port)
    if assessment.classification is HostClass.PUBLIC:
        return
    if assessment.classification is HostClass.FORBIDDEN:
        msg = f"fetch blocked forbidden host {parsed.host!r} (addresses: {', '.join(assessment.addresses)})"
        raise FetchPolicyError(msg)

    # PRIVATE
    if has_private_grant(parsed.host, parsed.port, instance=inst):
        return

    from plyngent.tools.workspace import get_policy_confirm_timeout

    timeout = (
        float(policy_timeout_seconds) if policy_timeout_seconds is not None else float(get_policy_confirm_timeout())
    )
    hook = get_fetch_policy_confirm_hook(inst)
    if hook is None:
        msg = (
            f"fetch blocked private/loopback host {parsed.host}:{parsed.port} "
            f"({assessment.reason}; no policy confirm hook / non-interactive deny)"
        )
        raise FetchPolicyError(msg)
    try:
        allowed = bool(hook(parsed.host, parsed.port, parsed.url, timeout))
    except Exception as exc:
        msg = f"fetch blocked private/loopback host {parsed.host}:{parsed.port} (confirm failed: {exc})"
        raise FetchPolicyError(msg) from exc
    if not allowed:
        msg = (
            f"fetch blocked private/loopback host {parsed.host}:{parsed.port} "
            f"(user declined or timed out after {timeout:g}s; not skipped by YOLO)"
        )
        raise FetchPolicyError(msg)
    _ = grant_private_host(parsed.host, parsed.port, instance=inst)


def format_grant_preview(host: str, port: int, url: str) -> Sequence[str]:
    return (
        f"host: {host}",
        f"port: {port}",
        f"url: {url}",
        f"grant_key: {grant_key(host, port)}",
    )
