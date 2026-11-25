"""A plugin claiming the ``fs`` group, which this program already has: one of the two is never reached."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import ToolSpec


class Files(ToolProvider):
    """A group named after one the program has, to see whether the checks say so."""

    @property
    @override
    def group(self) -> str:
        """The name of a group this program already answers to."""
        return "fs"

    @property
    @override
    def summary(self) -> str:
        """What this group would be, in a phrase nobody ever reads."""
        return "some files"

    @property
    @override
    def guide(self) -> str:
        """Nothing, because a session asking for this name is answered by the program's own group."""
        return "nothing, because this group is never reached."

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up: nothing to start."""
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """No tools at all: the name is the whole of what this file is about."""
        return ()

    @override
    async def close(self) -> None:
        """Nothing is held, so there is nothing to let go of."""


def enable() -> tuple[Declaration, ...]:
    """The declarations of a plugin that cannot be reached once it is up."""
    return (Declaration(ToolSet, instance="files", kwargs={"providers": (Files(),)}),)
