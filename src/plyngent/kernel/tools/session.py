"""The conversation's own two subjects: the plan as a stack of tasks, and the person at the other end.

One value, asked for, added to, and changed a task at a time. ``todo_list`` is the only one that shows
a window of the stack; a call that changes one layer answers **that layer**, which is why only the
listing takes a ``full``: there is nothing to open a window on when the call was about one group in the
first place.

``wait`` is the other half of the group, and the only call that spends time and touches no state: a person
reading what was just written needs a moment, and a turn that ended cannot be picked up again.

The stack belongs to the conversation and not to this process: every change is written into the
session's own history as one state change (§10.9, and ``Session.change``), so reopening a conversation
brings its plan back, one answer to a time. What that costs is a lock — two calls in one answer really
do run at once, and both would otherwise push onto the stack they read before either wrote.

Two numbers address a task, and the schema says so because the schema is the only place a model can
read a rule before it calls: ``group`` is a depth (0 is the bottom, the TOP group is the highest) and
``number`` is the task's place inside that group. Both come from the latest listing, and both are
refused, never guessed at, when they name nothing.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, cast

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.notices import PLAN
from plyngent.kernel.session import TODOS
from plyngent.kernel.todos import OPEN, Todos, TodoStatus
from plyngent.kernel.toolresult import Hint, Meta, Part, Text, ToolResult
from plyngent.kernel.toolspec import ArgsError, Group, Mutation, Res, ToolArgs, ToolSpec, argument
from plyngent.kernel.words import counted, plural
from plyngent.wire.ask import FreeText, as_written

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.kernel.env import Env
    from plyngent.kernel.session import Session


class ListArgs(ToolArgs, kw_only=True):
    """List the todo stack: the TOP group by default, every group with ``full=true``.

    The stack is a LIFO of groups: one push is one level of breakdown, and the top group is the part
    being worked on now. Groups are numbered by depth — ``Group 0`` is the bottom and the top group is
    numbered last and marked ``(TOP)`` — so pushing another group never renames the ones under it.
    Items are numbered inside their group from 1, which is the order the listing prints them in: a
    group's depth and a task's number are what ``todo_update`` takes.
    """

    full: bool = False


class PushArgs(ToolArgs, kw_only=True):
    """Push one new group of sibling tasks onto the top of the stack.

    ``titles`` is the group: one call is one level of breakdown, so several titles are siblings and not
    several levels. Blank titles are dropped, and nothing is pushed when every one of them was blank.
    ``notes`` is written onto every member of the new group. A stack already as deep as this program goes
    refuses the push: finish the TOP group and pop it first.
    """

    titles: Annotated[list[str], msgspec.Meta(min_length=1)]
    notes: str = ""


#: What a deferred pause answers with: what just happened, and what to spend this turn on. It never asks
#: the model to watch for a turn boundary, because nothing in the conversation marks one — the recording is
#: the kernel's to make, and the boundary is its own to see (E188).
DEFERRED = (
    "held for the next turn: group {group} item {number} is recorded when the harness starts one itself. "
    "Work on what else is open, and say so if there is nothing."
)


class UpdateArgs(ToolArgs, kw_only=True):
    """Change one task: its status, its title, its notes, or several at once.

    ``group`` is the depth the listing printed (0 is the bottom, the TOP group is the highest) and
    ``number`` is the task's number inside that group. Both come from the latest ``todo_list``, and
    nothing is changed when a number names nothing. A field left out is a field this call does not
    touch; ``notes=""`` clears a note. ``status`` is one of ``pending``, ``in_progress``, ``done``,
    ``cancelled``, ``paused``. ``paused`` means the task cannot be done now: it is recorded by the
    harness at the next turn it starts itself, so this call answers with that and writes nothing.
    """

    group: Annotated[int, msgspec.Meta(ge=0)]
    number: Annotated[int, msgspec.Meta(ge=1)]
    status: TodoStatus | None = None
    title: str | None = None
    notes: str | None = None


class PopArgs(ToolArgs, kw_only=True):
    """Pop the TOP group off the stack, and answer the group that becomes the top one.

    A pop takes the whole level back off: the tasks in it are not finished one by one by being
    removed, which is what ``todo_update`` is for. An empty stack is not an error — it is a stack with
    nothing on it.
    """


class ClearArgs(ToolArgs, kw_only=True):
    """Empty the stack: every group and every task, gone in one call.

    The plan is the conversation's, so clearing it clears it for good; what was there is still in the
    history as the state changes that wrote it, one entry at a time.
    """


# -- waiting for a person -------------------------------------------------------------------------


#: The longest a ``wait`` may sleep. In the schema, so a call that asks for an hour is one the model can
#: see is wrong before it makes it, and bounded because past this a wait is a turn that never finishes.
MAX_WAIT_SECONDS = 600.0


class WaitArgs(ToolArgs, kw_only=True):
    """Wait for a person to say something, for up to ``seconds``.

    The answer has three endings, and they are the three things a person can do: write a line, press
    Enter on an empty one, or say nothing until the wait runs out. Waiting is the one thing a
    conversation cannot do for itself — somebody reading what was just written needs a moment, and a turn
    that ended has no way to be picked up again — so it is a call, and the kernel keeps waiting.
    """

    seconds: Annotated[float, msgspec.Meta(gt=0, le=MAX_WAIT_SECONDS)]


async def wait(env: Env, args: WaitArgs) -> ToolResult:
    """Sleep until somebody speaks, and say which of the three things happened."""
    limit = _duration(args.seconds)
    if not env.tools.asks:
        # Nobody to ask is not nothing to wait for: a run with no terminal still sleeps. The sleep is
        # inside the call's own scope, so the cancellation a person sends from outside still stops it.
        await asyncio.sleep(args.seconds)
        return ToolResult.ok(f"waited {limit}: nobody is at a terminal to say anything")
    answer = as_written(
        await env.ask(FreeText(prompt=f"waiting up to {limit} — {args.intention}", timeout=args.seconds), at="wait")
    )
    match answer.text:
        case None:
            return ToolResult.ok(f"nobody spoke in {limit}")
        case "":
            return ToolResult.ok("the person pressed enter on an empty line")
        case words:
            return ToolResult.ok(f"the person wrote: {words}")


def _duration(seconds: float) -> str:
    """A number of seconds written the way a summary line says it: ``2.5s``, not ``2.500000s``."""
    return f"{seconds:g}s"


async def _planned(env: Env, mutate: Callable[[str | None], tuple[str | None, ToolResult]]) -> ToolResult:
    """Change the plan, and leave the model a line about what is left of it.

    The plan is the one setting a model has to hear about. It lives in the session's own store, outside
    the conversation, so a checklist written two rounds ago is one the model goes on working from memory —
    and the round after the call is exactly when it needs telling. So every change fires one reminder,
    and the turn writes it at its next boundary, which is that round (E142).

    The sentence is the stack's own (:meth:`~plyngent.kernel.todos.Todos.reminder`), and it is empty when
    nothing is open: that is how "the reminder you were given no longer holds" is said, and the summary
    beside it is the one thing a person can read in that case.
    """
    answer = await env.session_state.change(TODOS, mutate)
    left = Todos.load(env.session_state.setting(TODOS)).reminder()
    env.session_state.notices.say(PLAN, left, summary=left or "nothing is left on the plan")
    return answer


async def todo_list(env: Env, args: ListArgs) -> ToolResult:
    """The TOP group, or every group when the call asked for the whole stack."""
    todos = Todos.load(env.session_state.setting(TODOS))
    drawn = None if args.full else {todos.levels() - 1}
    shown = todos.levels() if drawn is None else 1
    hidden = todos.levels() - shown
    body: list[Part] = [Text(value=todos.render(drawn))]
    if hidden:
        body.append(_hidden(hidden))
    return ToolResult.ok(_counted(todos, shown=shown), body=body, meta=Meta(truncated=bool(hidden)))


async def todo_push(env: Env, args: PushArgs) -> ToolResult:
    """One new group of siblings, answered as the group it became."""

    def push(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        limit = env.session_state.host.depth
        if limit and todos.levels() >= limit:
            return None, _too_deep(todos, limit)
        group = todos.push_group(args.titles, notes=args.notes)
        summary = f"pushed {counted(len(group.items), 'item')} ({counted(todos.levels(), 'group')})"
        return todos.dump(), _layer(todos, todos.levels() - 1, summary)

    return await _planned(env, push)


async def todo_update(env: Env, args: UpdateArgs) -> ToolResult:
    """One task changed, answered as the group it belongs to."""

    if args.status is TodoStatus.PAUSED:
        return await _set_aside(env, args)

    def update(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        group = todos.group(args.group)
        if group is None:
            return None, _no_group(todos, args.group)
        if args.number > len(group.items):
            return None, _no_item(args.group, args.number, len(group.items))
        _ = todos.update(args.group, args.number, title=args.title, status=args.status, notes=args.notes)
        summary = f"updated group {args.group} item {args.number}{_changed(args)}"
        return todos.dump(), _layer(todos, args.group, summary)

    return await _planned(env, update)


async def _set_aside(env: Env, args: UpdateArgs) -> ToolResult:
    """Ask the plan to set one task aside, which is recorded at a turn the kernel starts (E188).

    Why not here, and why one turn: writing it now would empty the plan of its last open item, and a plan
    with nothing open is a chain that ends — so the request that just said "this cannot be done now" would
    be the thing that stopped the conversation from ever recording it. The turn between the asking and the
    recording is the price, and it is only asked for while the chain still has a turn to spend: with no
    turn left to protect, the pause is written at once.
    """
    session = env.session_state
    # A request from an earlier turn that never got its turn is recorded now, before this one is queued:
    # one slot, and the older promise is the one already made (E188).
    _ = await session.land_pause(turn=env.turn)
    todos = Todos.load(session.setting(TODOS))
    group = todos.group(args.group)
    if group is None:
        return _no_group(todos, args.group)
    if args.number > len(group.items):
        return _no_item(args.group, args.number, len(group.items))
    if _rounds_kept(session):
        asked = session.pause(env.turn, args.group, args.number)
        return ToolResult.of(Failure.NEXT_TURN, DEFERRED.format(group=asked.group, number=asked.number))

    def pause(before: str | None) -> tuple[str | None, ToolResult]:
        held = Todos.load(before)
        item = held.pause(args.group, args.number)
        summary = f"set aside group {args.group} item {args.number}"
        if item is None:
            return None, ToolResult.ok(f"group {args.group} item {args.number} was already set aside")
        return held.dump(), _layer(held, args.group, summary)

    return await _planned(env, pause)


def _rounds_kept(session: Session) -> bool:
    """Whether the chain this turn belongs to has another turn to spend, which is what a wait costs.

    Two things have to hold and they are the two the kernel checks before it starts a turn of its own: the
    switch is on, and this chain is under the number of turns a plan may drive. With neither, there is
    nothing to protect from a plan that empties — the pause is written immediately (E188).
    """
    limit = session.host.goals
    return session.goal and (not limit or session.goals < limit)


async def todo_pop(env: Env, _args: PopArgs) -> ToolResult:
    """The TOP group taken off, answered as the group that is now the top one."""

    def pop(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        group = todos.pop()
        if group is None:
            empty = ToolResult.ok(
                "nothing to pop: the stack is empty",
                body=Hint(tool=TODO_LIST.name, why="see what is on the stack"),
            )
            return None, empty
        summary = f"popped {counted(len(group.items), 'item')} ({counted(todos.levels(), 'group')})"
        return todos.dump(), _layer(todos, todos.levels() - 1, summary)

    return await _planned(env, pop)


async def todo_clear(env: Env, _args: ClearArgs) -> ToolResult:
    """Every group taken off at once, answered with the stack that is left."""

    def clear(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        count = todos.clear()
        answer = ToolResult.ok(f"cleared {counted(count, 'item')}", body=Text(value=todos.render()))
        return (todos.dump() if count else None), answer

    return await _planned(env, clear)


def _layer(todos: Todos, depth: int, summary: str) -> ToolResult:
    """One layer of the stack, and how much of the rest of it was not drawn beside it.

    The hint says "not shown", never "below", because a call that changed a middle layer has
    groups on both sides of the one it answered.
    """
    hidden = todos.levels() - 1
    body: list[Part] = [Text(value=todos.render({depth}))]
    if hidden:
        body.append(_hidden(hidden))
    return ToolResult.ok(summary, body=body, meta=Meta(truncated=bool(hidden)))


def _hidden(count: int) -> Hint:
    """The groups a result kept back, as the call that would show all of them."""
    return Hint(tool=TODO_LIST.name, given=(("full", "true"),), why=f"{count} more {plural(count, 'group')} not shown")


def _counted(todos: Todos, *, shown: int | None = None) -> str:
    """What the stack holds — in the window it was drawn in, when the call asked for a window."""
    groups = todos.levels()
    held = f"{shown} of {groups} groups" if shown is not None and shown != groups else counted(groups, "group")
    items = todos.items()
    if not items:
        return f"{held}, no items"
    opened = sum(1 for item in items if item.status in OPEN)
    paused = sum(1 for item in items if item.status is TodoStatus.PAUSED)
    tail = f", {paused} paused" if paused else ""
    return f"{held}, {counted(len(items), 'item')} ({opened} open{tail})"


def _changed(args: UpdateArgs) -> str:
    """What this call changed, as the words the answer is written in."""
    said = f" → {args.status.value}" if args.status is not None else ""
    named = [name for name in ("title", "notes") if argument(args, name) is not None]
    return f"{said} ({', '.join(named)})" if named else said


def _no_group(todos: Todos, depth: int) -> ToolResult:
    """A group number the stack does not have, and where numbers that do come from."""
    held = "the stack is empty" if not todos.levels() else f"the stack has {todos.levels()} ({_deepest(todos)})"
    return ToolResult.of(Failure.ARGS, f"no group {depth}: {held}", hints=(_listing_hint(),))


def _no_item(group: int, number: int, count: int) -> ToolResult:
    """An item number that group does not have."""
    return ToolResult.of(
        Failure.ARGS,
        f"no item {number} in group {group}: it has {counted(count, 'item')}",
        hints=(_listing_hint(),),
    )


def _listing_hint() -> Hint:
    """Where both numbers are written down, said once for both ways of getting them wrong."""
    return Hint(tool=TODO_LIST.name, why="both numbers come from the latest listing")


def _deepest(todos: Todos) -> str:
    """Which numbers the stack does have, without spelling out a range a model has to work out."""
    return "0 is the bottom" if todos.levels() == 1 else f"0 is the bottom, {todos.levels() - 1} is the top"


def _too_deep(todos: Todos, limit: int) -> ToolResult:
    """A push the stack has no room for, and the two ways a model can get room for it.

    What it protects is the one thing a stack of groups is for: the TOP group is the work being done now,
    and a breakdown deeper than this program allows has stopped naming any (E187).
    """
    return ToolResult.of(
        Failure.BLOCKED,
        f"the stack already has {counted(limit, 'group')}, which is as deep as this program goes ({_deepest(todos)})",
        hints=(
            Hint(tool=TODO_UPDATE.name, why="work the TOP group: it is the level being done now"),
            Hint(tool=TODO_POP.name, why="or finish that level and pop it, and this push will fit"),
        ),
    )


def _titles_given(args: Any) -> None:
    """A group has to say what is in it, or a push would be a level of breakdown that means nothing."""
    given = cast("list[object]", argument(args, "titles") or [])
    said = [str(title).strip() for title in given]
    if not any(said):
        msg = "give at least one title: a group whose every task is blank is not a level of breakdown"
        raise ArgsError("titles", msg)


def _something_to_change(args: Any) -> None:
    """A call that names no field changes nothing, and saying so is better than writing down nothing."""
    if all(argument(args, name) is None for name in ("status", "title", "notes")):
        msg = "give at least one of 'status', 'title' or 'notes': this call would change nothing"
        raise ArgsError("", msg)
    title = argument(args, "title")
    if title is not None and not str(title).strip():
        msg = "give a title that is not blank, or leave it out"
        raise ArgsError("title", msg)


TODO_LIST = ToolSpec(
    name="todo_list",
    group=Group.SESSION,
    args=ListArgs,
    run=todo_list,
    uses=Res.HISTORY,
)

TODO_PUSH = ToolSpec(
    name="todo_push",
    group=Group.SESSION,
    args=PushArgs,
    run=todo_push,
    uses=Res.HISTORY,
    mutates=Mutation.SESSION,
    guard=_titles_given,
)

TODO_UPDATE = ToolSpec(
    name="todo_update",
    group=Group.SESSION,
    args=UpdateArgs,
    run=todo_update,
    uses=Res.HISTORY,
    mutates=Mutation.SESSION,
    guard=_something_to_change,
)

TODO_POP = ToolSpec(
    name="todo_pop",
    group=Group.SESSION,
    args=PopArgs,
    run=todo_pop,
    uses=Res.HISTORY,
    mutates=Mutation.SESSION,
)

TODO_CLEAR = ToolSpec(
    name="todo_clear",
    group=Group.SESSION,
    args=ClearArgs,
    run=todo_clear,
    uses=Res.HISTORY,
    mutates=Mutation.SESSION,
)

#: The kernel's own waiting: a person's line with a deadline. ``TOOLS`` is declared beside ``HUMAN``
#: because a wait has to know whether anybody is there at all — ``env.tools.asks`` is that answer, and a
#: run with no terminal sleeps, and never asks a no-sink that answers at once.
WAIT = ToolSpec(
    name="wait",
    group=Group.SESSION,
    args=WaitArgs,
    run=wait,
    uses=Res.HUMAN | Res.TOOLS,
)

SPECS: tuple[ToolSpec, ...] = (TODO_LIST, TODO_PUSH, TODO_UPDATE, TODO_POP, TODO_CLEAR, WAIT)
