"""A plugin whose tool is called ``read``: the program has that name, so one of the two never runs."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


class ReadArgs(ToolArgs, kw_only=True):
    """The arguments of a tool that is not this one."""

    path: str = ""


async def read_own(_env: Env, _args: ReadArgs) -> ToolResult:
    """Never reached: the program's own ``read`` is what a model gets."""
    return ToolResult.ok("nothing was read, because this tool is never the one that runs")


READ = ToolSpec(
    name="read",
    group="taken",
    args=ReadArgs,
    run=read_own,
    uses=Res.PATHS,
    mutates=Mutation.NONE,
)


class Taken(ToolProvider):
    """A group of one, whose one name is the program's already."""

    @property
    @override
    def group(self) -> str:
        """A name of its own, so the tool name is the only collision here."""
        return "taken"

    @property
    @override
    def summary(self) -> str:
        """What this group claims to be."""
        return "a name that is taken"

    @property
    @override
    def guide(self) -> str:
        """Nothing, because the name belongs to the program."""
        return "nothing, because the name belongs to the program."

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up: nothing to start."""
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """The one tool, whose name is already answered for."""
        return (READ,)

    @override
    async def close(self) -> None:
        """Nothing is held, so there is nothing to let go of."""
        return


def enable() -> tuple[Declaration, ...]:
    """The declarations of a plugin whose tool can never be the one that answers."""
    return (Declaration(ToolSet, instance="taken", kwargs={"providers": (Taken(),)}),)
