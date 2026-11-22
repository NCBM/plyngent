"""The one tool about the tool set itself: asking to be shown a group (§10.4).

A session starts with the residents and gains a group by asking, and this is the asking. It is the
only tool that talks about the tools themselves, which is why it is the only member of
``Group.META`` — and why it has to be resident: a hint that points at a group nobody has been shown
is rewritten to a call of this tool, and a fallback nobody can reach is not a fallback.

It is also the only place that may ask for a group that does not exist yet: a *provider*'s group has no
tools until somebody wants them, and "wanting" is this call. That is one door for both kinds of group
on purpose — a model asks for a name it was told about, and whether the program already had the tools
is not its business (E175).

What it says is what arrived, not what exists: the tools take effect from the next round, because the
block a model is given may not change inside a round, so this answer is the one place the names can be
said before they are in the request. Asking twice for one group costs one change of the tool set —
the second call finds nothing new and says so, which is what keeps two calls in one answer from
rewriting the prompt prefix twice (§10.5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.toolspec import Group, Mutation, Res, ToolArgs, ToolSpec
from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


class RequireArgs(ToolArgs, kw_only=True):
    """Show this session the tools of one group, from the next round on.

    ``group`` is the name the index line lists — ``document``, ``fs``, ``process`` — and not a tool
    name: a group is what arrives together, with the section of the playbook that says how to use it.
    The set of names is not in the schema because it is a fact about this build's registry: a schema
    that moved with the registry would rewrite the prompt prefix every time a tool was added (§10.2).
    """

    group: Annotated[str, msgspec.Meta(min_length=1, max_length=64)]


async def tool_require(env: Env, args: RequireArgs) -> ToolResult:
    """Show one group, and answer the tools that were not reachable before this call.

    Two kinds of group arrive through here, and they are the same request from a model's side. A group of
    the program's own already has tools and is only being shown. A provider's group has no tools at all
    until somebody asks, so the asking *is* the making — which is why this is the only place that may
    ask, and why the words the provider wants said arrive with the answer rather than in the prompt
    (E175).
    """
    view = env.tools
    members = dict(view.groups).get(args.group)
    if members is None:
        if args.group in {group for group, _summary in view.available}:
            return await _arriving(env, args.group)
        # Only the groups with something left in them: a name the index line never prints is not a name
        # worth teaching a model, and a group whose tools are all resident has nothing to give.
        waiting = [group for group, names in view.groups if any(view.get(name) is None for name in names)]
        can = [group for group, _summary in view.available]
        return ToolResult.of(Failure.ARGS, f"no group named {args.group!r}; there are: {', '.join([*waiting, *can])}")
    # Asked before the disclosure, so "not visible" means "arriving now": the view a call is handed was
    # built for the round it runs in, and the group it is about to add is exactly what it does not hold.
    arriving = [name for name in members if view.get(name) is None]
    if not await env.session_state.disclose([args.group]):
        return ToolResult.ok(f"{args.group} was already shown")
    if not arriving:
        return ToolResult.ok(f"{args.group} has nothing left to show: its tools are resident already")
    return ToolResult.ok(f"showing {counted(len(arriving), 'tool')} from {args.group}: {', '.join(arriving)}")


async def _arriving(env: Env, group: str) -> ToolResult:
    """Ask this process for a group nothing has made yet, then show it and say how to use it.

    A provider that could not come up is a result and not a bug: somebody else's server may be down, and
    the sentence says so and says to ask again — the same call, because nothing about the arguments was
    wrong. Its deadline is the provider's own, spent in one place, so what arrives here is either the
    group or an answer about why not.
    """
    try:
        arrived = await env.materialise(group)
    except Cancelled:
        raise
    except TimeoutError:
        return ToolResult.of(Failure.TIMEOUT, f"{group} did not come up in its time; ask again if you still need it")
    except Exception as exc:  # noqa: BLE001 — a provider that failed is a result, not a stopped turn
        return ToolResult.of(Failure.BUSY, f"{group} could not be brought up: {exc}. Ask again if you still need it")
    shown = await env.session_state.disclose([group])
    if not shown:
        # The group was disclosed in an earlier process or an earlier round, so the row that shows it is
        # already in the log — but the tool block moved just now, and the count is what says so (E175).
        _ = await env.session_state.notice_tools()
    _ = await env.session_state.host.trail.snapshot(env.session)
    names = ", ".join(spec.name for spec in arrived.specs)
    said = f"{group} is up: {counted(len(arrived.specs), 'tool')}"
    if names:
        said += f" ({names})"
    return ToolResult.ok(said if not arrived.guide else f"{said}\n\n{arrived.guide}")


TOOL_REQUIRE = ToolSpec(
    name="tool_require",
    group=Group.META,
    args=RequireArgs,
    run=tool_require,
    uses=Res.TOOLS | Res.HISTORY,
    mutates=Mutation.SESSION,
)

SPECS: tuple[ToolSpec, ...] = (TOOL_REQUIRE,)
