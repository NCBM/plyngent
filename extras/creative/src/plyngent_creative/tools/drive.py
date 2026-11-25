"""The three calls that drive a plugin through a real turn.

Where ``plugin_check`` reads a file, these run it: ``plugin_test`` starts a kernel in a child process and
carries out a script, ``plugin_debug`` carries on from where that stopped, and ``plugin_stop`` finishes and
takes the root away. A failed step does not end anything — the handle is still good, which is what makes
"fix the file, ask again" one more call (E184).

**A root of its own, and it is granted.** The child's kernel is brought up over a root under the system temp
dir with its journal beside it, so nothing a plugin does reaches a tree a person works in. That root is
granted to the conversation that asked, which is what makes it readable with ``read`` and changeable with
``write_file`` while the run is open: debugging a plugin is looking at what it did, and that is a path
somebody already has tools for (E152, E184).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

import msgspec

from plyngent.kernel.access import AccessMode
from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import Hint, Text, ToolResult
from plyngent.kernel.toolspec import Mutation, Res, ToolArgs, ToolSpec
from plyngent_creative.steps import ScriptError, Step, parse
from plyngent_creative.tools.group import GROUP
from plyngent_creative.tools.plugins import NAME_PATTERN, PluginNameError, path_of, prepared
from plyngent_creative.tools.runs import RUNS, Driven, NoHandleError

if TYPE_CHECKING:
    from plyngent.kernel.env import Env

#: How many lines of a transcript a report prints. A run can be long, and what an author reads first is the
#: last few steps and the events under them.
_SHOWN = 20


class RunArgs(ToolArgs, kw_only=True):
    """Run a script against a plugin, in a kernel of its own, and answer a handle to what happened.

    ``name`` is the plugin, named the way ``plugin_check`` names it. ``steps`` is the script: a linear
    sequence of static JSON, each step saying what the model answers, what is sent, and what has to arrive —
    ``plugin_guide`` is the page about the three keys and the order they happen in.

    The kernel is a real one, in a child process, over a root under the system temp dir. That root is granted
    to this conversation, so ``read`` and ``write_file`` can look at what the plugin did while the run is
    still open. ``plugin_debug`` carries on from the handle, and ``plugin_stop`` finishes.
    """

    name: Annotated[str, msgspec.Meta(pattern=NAME_PATTERN)]
    #: A step is free-form JSON on purpose: its three keys are read by a grammar that answers sentences about
    #: them (``plugin_guide`` is the page), and a schema that pinned them here would be a second spelling of
    #: that grammar to keep in step (E184).
    steps: Annotated[
        list[dict[str, Any]],
        msgspec.Meta(
            description=(
                "a linear sequence of steps; each says `model` (what the scripted model answers), `send` "
                "(a line, or a command object with a `cmd`), and `expect` ({tag, fields}). `plugin_guide` "
                "has the whole grammar."
            )
        ),
    ]


async def plugin_test(env: Env, args: RunArgs) -> ToolResult:
    """Start a run, carry out the script, and answer the handle and what came of it."""
    root = await prepared(env)
    try:
        found = path_of(args.name, where=root)
    except PluginNameError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    if not found.is_file():
        return ToolResult.of(Failure.MISSING, f"no plugin named {args.name!r} yet: {found} is where one goes")
    try:
        steps = parse(args.steps)
    except ScriptError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    try:
        run = await RUNS.open(found)
    except NoHandleError as exc:
        return ToolResult.of(Failure.TOOL, f"a run of {args.name!r} could not be started: {exc}")
    return await carried(env, run, steps, first=True)


class DebugArgs(ToolArgs, kw_only=True):
    """Carry on with a run, and answer what happened this time.

    ``handle`` is what ``plugin_test`` answered with, and ``steps`` is what to do next — an empty list is
    "read out everything that has arrived since the last call", which is how a run that is waiting on a turn
    is looked at without typing anything into it.

    A step that failed is a sentence and not the end: the run stays where it is, so a plugin that was fixed
    in between is one more call away from being tried again.
    """

    handle: int
    #: The steps to carry out now, in the same grammar ``plugin_test`` takes. Empty reads out what arrived.
    steps: Annotated[list[dict[str, Any]], msgspec.Meta(description="the next steps, or empty to read out")]


async def plugin_debug(env: Env, args: DebugArgs) -> ToolResult:
    """Carry out more steps on the run one handle names, or read out what is already there."""
    try:
        run = RUNS.get(args.handle)
    except NoHandleError as exc:
        return _none(exc)
    try:
        steps = parse(args.steps)
    except ScriptError as exc:
        return ToolResult.of(Failure.ARGS, str(exc))
    return await carried(env, run, steps, first=False)


class StopArgs(ToolArgs, kw_only=True):
    """Finish with a run: stop its process, take its root away, and answer what it did.

    ``handle`` is what ``plugin_test`` answered with. A handle that is already gone is answered with what it
    means — there is no run — so a person who calls this twice is told the truth both times.
    """

    handle: int


async def plugin_stop(_env: Env, args: StopArgs) -> ToolResult:
    """Stop one run, and say what it had done."""
    try:
        run = await RUNS.close(args.handle)
    except NoHandleError as exc:
        return _none(exc)
    lines = [*_transcript(run), "", f"the root {run.root} is gone"]
    return ToolResult.ok(
        f"stopped run {args.handle} of {run.plugin.stem}: {len(run.answers)} steps, {len(run.events)} events",
        body=Text(value="\n".join(lines)),
    )


async def carried(env: Env, run: Driven, steps: tuple[Step, ...], *, first: bool) -> ToolResult:
    """Carry out the steps this call was given, one at a time, and answer what came of them.

    One at a time and not all at once: a step is the unit an author fixes, so the answer says which one went
    wrong and what had already happened. The run is left where the first failure stopped it, because a failed
    step is a sentence and not the end of the run (E184).
    """
    _ = await env.grant(run.root, AccessMode.READ)
    before = len(run.answers)
    wrong: str | None = None
    for one in steps:
        answer = await run.step(one)
        if answer != "ok":
            wrong = answer
            break
    done = len(run.answers) - before
    label = f"started run {run.handle}" if first else f"run {run.handle}"
    summary = f"{label}: {done} of {len(steps)} steps done" if steps else f"{label}: nothing new"
    told = Text(value="\n".join(_transcript(run)))
    hint = Hint(tool=DEBUG.name, given=(("handle", str(run.handle)),), why="what has arrived, or the next steps")
    if wrong is not None:
        return ToolResult.of(Failure.TOOL, f"{summary}; {wrong}", body=(told, hint))
    return ToolResult.ok(summary, body=(told, hint))


def _transcript(run: Driven) -> list[str]:
    """What the run has said, as lines a person reads: one per step, and the events under them."""
    lines: list[str] = []
    for at, answer in enumerate(run.answers[-_SHOWN:], start=max(1, len(run.answers) - _SHOWN + 1)):
        lines.append(f"step {at}: {answer}")
    if run.events:
        lines.append(f"{len(run.events)} events, the last {_SHOWN}:")
        lines.extend(f"  {_event(one)}" for one in run.events[-_SHOWN:])
    if run.tail:
        lines.append("what it wrote to its error stream:")
        lines.extend(f"  {line}" for line in run.tail.splitlines()[-_SHOWN:])
    return lines


def _event(one: dict[str, object]) -> str:
    """One event as a line: its tag, and the fields the wire carries that a line can hold.

    The tag is left out of the fields, because it is the word the line already begins with: the wire spells it
    twice — once as the union's tag and once inside the object — and a line that said it twice would be one
    fact printed twice.
    """
    fields = one.get("fields")
    held = cast("dict[str, object]", fields) if isinstance(fields, dict) else {}
    plain = [
        f"{name}={value!r}"
        for name, value in held.items()
        if name != "event" and isinstance(value, (str, int, float, bool))
    ]
    return " ".join([str(one.get("tag")), *plain])


def _none(why: Exception) -> ToolResult:
    """The answer to a handle nothing is open under, which is the same for both calls that take one."""
    return ToolResult.of(Failure.ARGS, str(why), hints=(Hint(tool=TEST.name, why="start a run of one"),))


TEST = ToolSpec(
    name="plugin_test",
    group=GROUP,
    args=RunArgs,
    run=plugin_test,
    # ``EXEC`` because a child process is started, and ``PATHS`` because its root is granted to this
    # conversation and resolved like any other path. What the child does is somebody else's process, which is
    # why ``OUTSIDE`` is declared and the posture that follows from it asks every time.
    uses=Res.PATHS | Res.EXEC,
    mutates=Mutation.OUTSIDE,
    subject="name",
)

DEBUG = ToolSpec(
    name="plugin_debug",
    group=GROUP,
    args=DebugArgs,
    run=plugin_debug,
    uses=Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

STOP = ToolSpec(
    name="plugin_stop",
    group=GROUP,
    args=StopArgs,
    run=plugin_stop,
    uses=Res.PATHS,
    mutates=Mutation.OUTSIDE,
)

SPECS: tuple[ToolSpec, ...] = (TEST, DEBUG, STOP)
