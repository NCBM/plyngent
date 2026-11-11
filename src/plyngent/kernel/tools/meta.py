"""The one tool about the tool set itself: asking to be shown a group (§10.4).

A session starts with the residents and gains a group by asking, and this is the asking. It is the
only tool whose subject is the tools rather than the world, which is why it is the only member of
``Group.META`` — and why it has to be resident: a hint that points at a group nobody has been shown
is rewritten to a call of this tool, and a fallback nobody can reach is not a fallback.

What it says is what arrived, not what exists: the tools take effect from the next round, because the
block a model is given may not change inside a round, so this answer is the one place the names can be
said before they are in the request. Asking twice for one group costs one change of the tool set —
the second call finds nothing new and says so, which is what keeps two calls in one answer from
rewriting the prompt prefix twice (§10.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Group, Mutation, Res, ToolArgs, ToolSpec

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


class RequireArgs(ToolArgs, kw_only=True):
    """Show this session the tools of one group, from the next round on.

    ``group`` is the name the index line lists — ``document``, ``fs``, ``process`` — and not a tool
    name: a group is what arrives together, with the section of the playbook that says how to use it.
    The set of names is not in the schema because it is a fact about this build's registry rather than
    about this tool, and a schema that moved with the registry would rewrite the prompt prefix every
    time a tool was added (§10.2).
    """

    group: Annotated[str, msgspec.Meta(min_length=1, max_length=64)]


async def tool_require(env: Env, args: RequireArgs) -> ToolResult:
    """Show one group, and answer the tools that were not reachable before this call."""
    members = dict(env.tools.groups).get(args.group)
    if members is None:
        # Only the groups with something left in them: a name the index line never prints is not a name
        # worth teaching a model, and a group whose tools are all resident has nothing to give.
        waiting = [group for group, names in env.tools.groups if any(env.tools.get(name) is None for name in names)]
        return ToolResult.of(Failure.ARGS, f"no group named {args.group!r}; there are: {', '.join(waiting)}")
    # Asked before the disclosure, so "not visible" means "arriving now": the view a call is handed was
    # built for the round it runs in, and the group it is about to add is exactly what it does not hold.
    arriving = [name for name in members if env.tools.get(name) is None]
    if not await env.session_state.disclose([args.group]):
        return ToolResult.ok(f"{args.group} was already shown")
    if not arriving:
        return ToolResult.ok(f"{args.group} has nothing left to show: its tools are resident already")
    return ToolResult.ok(f"showing {len(arriving)} tools from {args.group}: {', '.join(arriving)}")


TOOL_REQUIRE = ToolSpec(
    name="tool_require",
    group=Group.META,
    args=RequireArgs,
    run=tool_require,
    uses=Res.TOOLS | Res.HISTORY,
    mutates=Mutation.SESSION,
)

SPECS: tuple[ToolSpec, ...] = (TOOL_REQUIRE,)
