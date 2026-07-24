"""Soft-confirm trust grants (session-scoped)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plyngent.tools.context import SessionState


def grant_key(tool_name: str) -> str:
    """Phase 1 grant key is the tool name (session isolation via SessionState)."""
    return tool_name


def has_grant(session: SessionState, tool_name: str) -> bool:
    return session.has_grant(grant_key(tool_name))


def add_grant(session: SessionState, tool_name: str) -> None:
    session.add_grant(grant_key(tool_name))


def clear_grants(session: SessionState) -> None:
    session.clear_grants()
