"""A plugin whose provider raises when it is asked for its tools: it never comes up, and nothing says why."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import ToolSpec


class Loud(ToolProvider):
    """A group that cannot be brought up, which is the one kind of trouble a check has to survive."""

    @property
    @override
    def group(self) -> str:
        """A name of its own, so nothing about the name is what goes wrong."""
        return "loud"

    @property
    @override
    def summary(self) -> str:
        """What this group claims to be, which it never gets to be."""
        return "a group that will not start"

    @property
    @override
    def guide(self) -> str:
        """Nothing, because this group raises before it has tools to describe."""
        return "nothing, because this group raises before it has tools."

    @property
    @override
    def timeout(self) -> float:
        """How long the group claims it may take: it fails at once, and the number never matters."""
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """The one thing wrong here: the program this group would have needed is not installed."""
        msg = "the program this group needed is not installed"
        raise RuntimeError(msg)

    @override
    async def close(self) -> None:
        """Nothing was started, so there is nothing to stop."""
        return


def enable() -> tuple[Declaration, ...]:
    """The declarations of a plugin that is broken the moment somebody asks for its tools."""
    return (Declaration(ToolSet, instance="loud", kwargs={"providers": (Loud(),)}),)
