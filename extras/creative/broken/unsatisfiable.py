"""A plugin whose schema requires a field it never declares: no argument package could ever satisfy it."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import ExternalArgs, Mutation, Res, ToolSpec

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


#: A schema with a requirement and no properties: the field it needs is one no caller can send.
UNSATISFIABLE = ExternalArgs(
    schema={
        "type": "object",
        "properties": {},
        "required": ["who"],
    },
    description="asks for a field that is not in the package",
)


async def shout(_env: Env, _args: object) -> ToolResult:
    """Never reached with a valid package, because no package is valid."""
    return ToolResult.ok("nothing was shouted, because no call can be made")


SHOUT = ToolSpec(
    name="shout",
    group="shouts",
    args=UNSATISFIABLE,
    run=shout,
    uses=Res(0),
    mutates=Mutation.NONE,
)


class Shouts(ToolProvider):
    """A group of one tool whose schema cannot be satisfied."""

    @property
    @override
    def group(self) -> str:
        """A name of its own, so the schema is the only thing wrong here."""
        return "shouts"

    @property
    @override
    def summary(self) -> str:
        """What this group claims to be."""
        return "a tool no call can make"

    @property
    @override
    def guide(self) -> str:
        """Nothing, because every call of this tool is refused for a field it cannot fill."""
        return "nothing, because every call of this tool is refused for a field it cannot fill."

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up: nothing to start."""
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """The one tool, whose schema no package satisfies."""
        return (SHOUT,)

    @override
    async def close(self) -> None:
        """Nothing is held, so there is nothing to let go of."""
        return


def enable() -> tuple[Declaration, ...]:
    """The declarations of a plugin whose one tool requires a field its schema never declares."""
    return (Declaration(ToolSet, instance="shouts", kwargs={"providers": (Shouts(),)}),)
