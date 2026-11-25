"""A plugin with declarations and no tool set in them: it says something and brings nothing to call."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from composat import Declaration


def enable() -> tuple[Declaration, ...]:
    """Nothing a session could ever reach."""
    return ()
