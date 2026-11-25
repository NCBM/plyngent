"""A plugin that is right, and the shape of every plugin: a tuple of declarations, a tool set of providers.

``Greets`` is one provider: it says which group it brings, what that group is for, and — when somebody asks
— the tools themselves. The group is not in the table until a conversation asks for it, which is what makes
it a provider rather than a tool set this program started with (E175).

Copy this file, rename the group and the tool, and the checks tell you what else has to be true.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from composat import Declaration

from plyngent.kernel.providing import ToolProvider, ToolSet
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


class GreetArgs(ToolArgs, kw_only=True):
    """Who to greet: ``ToolArgs`` is where the one field every call may carry is declared."""

    name: str = "world"


async def greet(_env: Env, args: GreetArgs) -> ToolResult:
    """The one tool this group brings: it says a sentence and changes nothing."""
    return ToolResult.ok(f"hello {args.name}")


GREET = ToolSpec(
    name="greet",
    group="greets",
    args=GreetArgs,
    run=greet,
    uses=Res(0),
    mutates=Mutation.NONE,
)


class Greets(ToolProvider):
    """The group, and the tools it has once somebody asks for it."""

    @property
    @override
    def group(self) -> str:
        """The name a session asks for, and the name the index line prints."""
        return "greets"

    @property
    @override
    def summary(self) -> str:
        """What this group is, in a phrase."""
        return "a greeting"

    @property
    @override
    def guide(self) -> str:
        """What a model is told about these tools, said when it asks for them."""
        return "`greet` says hello to one name and changes nothing."

    @property
    @override
    def timeout(self) -> float:
        """How long this group may take to come up: nothing to start, so nothing to wait for."""
        return 5.0

    @override
    async def materialise(self) -> tuple[ToolSpec, ...]:
        """The tools that are up now, which for this group is a table it always had."""
        return (GREET,)

    @override
    async def close(self) -> None:
        """Nothing is held, so there is nothing to let go of."""


def enable() -> tuple[Declaration, ...]:
    """What ``--plugin-file`` calls: the declarations this plugin brings."""
    return (Declaration(ToolSet, instance="greets", kwargs={"providers": (Greets(),)}),)
