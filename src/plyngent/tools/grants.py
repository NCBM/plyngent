"""Soft-confirm trust grants (session-scoped).

Live map: ``SessionState.grants`` (sync reads for the confirm gate).
Durable tree: ``session.data["grants"]`` (tool name → bool) via
:class:`~plyngent.tools.view.PersistentDataView`.

Hosts that resume a session document should call :func:`hydrate_grants`
after binding :class:`~plyngent.tools.context.SessionState`. Soft-confirm
approval calls :func:`add_grant`, which updates the live map and commits
the view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plyngent.tools.context import SessionState


def grant_key(tool_name: str) -> str:
    """Phase 1 grant key is the tool name (session isolation via SessionState)."""
    return tool_name


def has_grant(session: SessionState, tool_name: str) -> bool:
    """Return whether *tool_name* is granted in the live session map."""
    return session.has_grant(grant_key(tool_name))


async def add_grant(session: SessionState, tool_name: str) -> None:
    """Grant *tool_name* and commit the grants map under ``session.data``."""
    session.add_grant(grant_key(tool_name))
    await persist_grants(session)


def clear_grants(session: SessionState) -> None:
    """Clear the live grant map (sync; e.g. CLI YOLO off / ``/new``).

    Does not open a view transaction (callers may be outside an event loop).
    Use :func:`persist_grants` when the durable tree should also clear.
    """
    session.clear_grants()


async def clear_grants_and_persist(session: SessionState) -> None:
    """Clear live grants and write an empty map to ``session.data``."""
    session.clear_grants()
    await persist_grants(session)


async def persist_grants(session: SessionState) -> None:
    """Write the live grants map to ``session.data["grants"]``."""
    async with session.data as data:
        data["grants"].store({key: bool(value) for key, value in session.grants.items()})


async def hydrate_grants(session: SessionState) -> None:
    """Load durable grants into the live map (replace)."""
    from typing import cast

    raw: object | None = None
    async with session.data as data:
        try:
            raw = data["grants"].load()
        except RuntimeError:
            raw = None
    session.grants.clear()
    if not isinstance(raw, dict):
        return
    blob = cast("dict[object, object]", raw)
    for key_obj, value in blob.items():
        if value:
            session.grants[str(key_obj)] = True
