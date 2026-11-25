"""The tool that runs a program: code-shaped tool calling, with control flow as the small part.

A conversation's cost is round trips, and a round trip is worth what the model learns from it. A task
that reads six files, keeps the two that match and prints them is one program and six calls, and the
model that writes the loop reads the answer instead of writing six answers (E190).

What a program is, and what it is not, is said in :class:`CodeArgs` because that docstring is the schema
the model reads; how to use one is said in the ``code`` group's playbook; which tools a program may call
is the ``tools`` argument, and the person who is asked about the program is asked about that list. The
three are one decision written three times for three readers, and none of them is allowed to grow a
fourth spelling here.

The plan is checked before anything runs (:mod:`plyngent.kernel.tools.code.plan`) and the program runs in
a process of its own (:mod:`plyngent.kernel.tools.code.sandbox`). This module is the order of those, and
the one question in between: a program is asked about every time, because what a person agrees to is the
text of the program and no yes to another program covers it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.toolresult import ToolResult
from plyngent.kernel.tools.code.plan import planned, reviewed, said
from plyngent.kernel.tools.code.sandbox import played
from plyngent.kernel.toolspec import Group, Mutation, Posture, Res, ToolArgs, ToolSpec
from plyngent.wire.ask import Confirmation, Severity, as_agreed

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.kernel.env import Env


class CodeArgs(ToolArgs, kw_only=True):
    """Run one Python program in which the tools named in ``tools`` are functions.

    A program is a value: it is not a file, it changes nothing about this conversation except through
    the tools it calls, and it answers only what it prints. ``print`` is the one way back — a step whose
    result is never printed is a step the conversation never reads, which is what makes a program cheaper
    than the calls it replaces. A call answers a ``Result``: ``str(r)`` is the text a person would read,
    ``r.ok`` and ``r.failure`` say how it went, and ``r.documents`` holds handles a later call can read.

    A call that did not work raises ``ToolFailed``, and the program decides what to do about that; one
    nobody catches ends the program with that failure, while the steps that already ran still stand.
    ``help()`` prints every tool this program may call, and ``help("read")`` prints one: its signature,
    which is generated from the schema, and the fields it takes.

    ``tools`` is the whole of what the program may call, by the names a round calls them by. A name
    outside it is refused before the program starts, and the answer names what it could call instead.
    """

    code: Annotated[str, msgspec.Meta(min_length=1, description="the program, as Python source")]
    tools: Annotated[list[str], msgspec.Meta(description="the tools this program may call, by name")]


async def run_toolscript(env: Env, args: CodeArgs) -> ToolResult:
    """Read the plan, read the program, ask a person about it, then run it.

    The order is the one that costs least: a name the plan cannot have and a shape this build will not run
    are both certain before anybody is asked, and a question about a program that would be refused is a
    question that taught somebody to say no to the next one.
    """
    plan = planned(env.tools, args.tools)
    if isinstance(plan, ToolResult):
        return plan
    refused = reviewed(args.code, plan)
    if refused is not None:
        return refused
    answer = as_agreed(
        await env.ask(
            Confirmation(prompt=_prompt(plan), severity=Severity.DANGER, executable_code=args.code),
            at="consent",
        )
    )
    if not answer.accepted:
        return ToolResult.unagreed(answer, about="running this program")
    return await played(env, code=args.code, plan=plan)


def _prompt(plan: Mapping[str, ToolSpec]) -> str:
    """What a person is asked about a program: what it may call, and what was checked about its text."""
    return (
        f"run a program that calls {said(plan)}? a static check found no dunder access in it, "
        "and every call it makes comes back here"
    )


TOOLSCRIPT = ToolSpec(
    name="run_toolscript",
    group=Group.CODE,
    args=CodeArgs,
    run=run_toolscript,
    uses=Res.EXEC | Res.PATHS | Res.DOCUMENT | Res.TOOLS,
    mutates=Mutation.OUTSIDE | Mutation.WORKSPACE,
    # The baseline already reads `AUTO` for a tool that runs programs, and it is said here because the
    # question is this handler's own: the person agreeing to a program is agreeing to its text, and the
    # load that carries it is in the question (§10.13, E191). The level is the baseline's too — anything
    # that runs a program is `DANGER` — and it is not spelled a second time.
    posture=Posture.AUTO,
)

SPECS: tuple[ToolSpec, ...] = (TOOLSCRIPT,)
