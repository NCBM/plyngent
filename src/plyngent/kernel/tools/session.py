"""The session's own plan, as tools: the todo stack (§10.9).

One value, asked for, added to, and changed a task at a time. ``todo_list`` is the only one that shows
a window of the stack; a call that changes one layer answers **that layer**, which is why only the
listing takes a ``full``: there is nothing to open a window on when the call was about one group in the
first place.

The stack belongs to the conversation and not to this process: every change is written into the
session's own history as one state change (§10.9, and ``Session.change``), so reopening a conversation
brings its plan back, one answer to a time. What that costs is a lock — two calls in one answer really
do run at once, and both would otherwise push onto the stack they read before either wrote.

Two numbers address a task, and the schema says so because the schema is the only place a model can
read a rule before it calls: ``group`` is a depth (0 is the bottom, the TOP group is the highest) and
``number`` is the task's place inside that group. Both come from the latest listing, and both are
refused rather than guessed when they name nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.session import TODOS
from plyngent.kernel.todos import OPEN, Todos, TodoStatus
from plyngent.kernel.toolresult import Hint, Meta, Part, Text, ToolResult
from plyngent.kernel.toolspec import ArgsError, Group, Mutation, Res, ToolArgs, ToolSpec, argument

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


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
    ``notes`` is written onto every member of the new group.
    """

    titles: Annotated[list[str], msgspec.Meta(min_length=1)]
    notes: str = ""


class UpdateArgs(ToolArgs, kw_only=True):
    """Change one task: its status, its title, its notes, or several at once.

    ``group`` is the depth the listing printed (0 is the bottom, the TOP group is the highest) and
    ``number`` is the task's number inside that group. Both come from the latest ``todo_list``, and
    nothing is changed when a number names nothing. A field left out is a field this call does not
    touch; ``notes=""`` clears a note. ``status`` is one of ``pending``, ``in_progress``, ``done``,
    ``cancelled``.
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


async def todo_list(env: Env, args: ListArgs) -> ToolResult:
    """The TOP group, or every group when the call asked for the whole stack."""
    todos = Todos.load(env.session_state.setting(TODOS))
    drawn = None if args.full else {todos.depth() - 1}
    shown = todos.depth() if drawn is None else 1
    hidden = todos.depth() - shown
    body: list[Part] = [Text(value=todos.render(drawn))]
    if hidden:
        body.append(_hidden(hidden))
    return ToolResult.ok(_counted(todos, shown=shown), body=body, meta=Meta(truncated=bool(hidden)))


async def todo_push(env: Env, args: PushArgs) -> ToolResult:
    """One new group of siblings, answered as the group it became."""

    def push(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        group = todos.push_group(args.titles, notes=args.notes)
        summary = f"pushed {_count(len(group.items), 'item')} (depth {todos.depth()})"
        return todos.dump(), _layer(todos, todos.depth() - 1, summary)

    return await env.session_state.change(TODOS, push)


async def todo_update(env: Env, args: UpdateArgs) -> ToolResult:
    """One task changed, answered as the group it belongs to."""

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

    return await env.session_state.change(TODOS, update)


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
        summary = f"popped {_count(len(group.items), 'item')} (depth {todos.depth()})"
        return todos.dump(), _layer(todos, todos.depth() - 1, summary)

    return await env.session_state.change(TODOS, pop)


async def todo_clear(env: Env, _args: ClearArgs) -> ToolResult:
    """Every group taken off at once, answered with the stack that is left."""

    def clear(before: str | None) -> tuple[str | None, ToolResult]:
        todos = Todos.load(before)
        count = todos.clear()
        answer = ToolResult.ok(f"cleared {_count(count, 'item')}", body=Text(value=todos.render()))
        return (todos.dump() if count else None), answer

    return await env.session_state.change(TODOS, clear)


def _layer(todos: Todos, depth: int, summary: str) -> ToolResult:
    """One layer of the stack, and how much of the rest of it was not drawn beside it.

    The hint says "not shown" rather than "below", because a call that changed a middle layer has
    groups on both sides of the one it answered.
    """
    hidden = todos.depth() - 1
    body: list[Part] = [Text(value=todos.render({depth}))]
    if hidden:
        body.append(_hidden(hidden))
    return ToolResult.ok(summary, body=body, meta=Meta(truncated=bool(hidden)))


def _hidden(count: int) -> Hint:
    """The groups a result kept back, as the call that would show all of them."""
    return Hint(tool=TODO_LIST.name, given=(("full", "true"),), why=f"{count} more {_plural(count, 'group')} not shown")


def _counted(todos: Todos, *, shown: int | None = None) -> str:
    """What the stack holds — in the window it was drawn in, when the call asked for a window."""
    groups = todos.depth()
    held = f"{shown} of {groups} groups" if shown is not None and shown != groups else _count(groups, "group")
    items = todos.items()
    if not items:
        return f"{held}, no items"
    opened = sum(1 for item in items if item.status in OPEN)
    return f"{held}, {_count(len(items), 'item')} ({opened} open)"


def _count(count: int, noun: str) -> str:
    """A number with its noun — "1 items" reads like a bug."""
    return f"{count} {_plural(count, noun)}"


def _changed(args: UpdateArgs) -> str:
    """What this call changed, as the words the answer is written in."""
    said = f" → {args.status.value}" if args.status is not None else ""
    named = [name for name in ("title", "notes") if argument(args, name) is not None]
    return f"{said} ({', '.join(named)})" if named else said


def _no_group(todos: Todos, depth: int) -> ToolResult:
    """A group number the stack does not have, and where numbers that do come from."""
    held = "the stack is empty" if not todos.depth() else f"the stack has {todos.depth()} ({_deepest(todos)})"
    return ToolResult.of(Failure.ARGS, f"no group {depth}: {held}", hints=(_listing_hint(),))


def _no_item(group: int, number: int, count: int) -> ToolResult:
    """An item number that group does not have."""
    return ToolResult.of(
        Failure.ARGS,
        f"no item {number} in group {group}: it has {_count(count, 'item')}",
        hints=(_listing_hint(),),
    )


def _listing_hint() -> Hint:
    """Where both numbers are written down, said once for both ways of getting them wrong."""
    return Hint(tool=TODO_LIST.name, why="both numbers come from the latest listing")


def _deepest(todos: Todos) -> str:
    """Which numbers the stack does have, without spelling out a range a model has to work out."""
    return "0 is the bottom" if todos.depth() == 1 else f"0 is the bottom, {todos.depth() - 1} is the top"


def _plural(count: int, noun: str) -> str:
    """The noun at the count it is used with, singular where one is one."""
    return noun if count == 1 else f"{noun}s"


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

SPECS: tuple[ToolSpec, ...] = (TODO_LIST, TODO_PUSH, TODO_UPDATE, TODO_POP, TODO_CLEAR)
