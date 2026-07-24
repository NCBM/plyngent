from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.agent import ToolTag, tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.agent.todo_stack import TodoStack, TodoStatus

_stack: TodoStack | None = None
_on_change: Callable[[], None] | None = None

_VALID_STATUS = frozenset({"pending", "in_progress", "done", "cancelled"})


def set_todo_stack(stack: TodoStack | None, *, on_change: Callable[[], None] | None = None) -> None:
    """Bind the session todo stack for tool handlers (and optional persist hook).

    Process-global bind remains for CLI/agent hosts that keep a live
    :class:`~plyngent.agent.todo_stack.TodoStack` outside the view store.
    Prefer session ``data["todo"]`` transactions when available.
    """
    global _stack, _on_change  # noqa: PLW0603 — intentional process bind
    _stack = stack
    _on_change = on_change


def get_todo_stack() -> TodoStack | None:
    return _stack


def _notify() -> None:
    """Fire host persist hooks: prefer session.on_todo_change, else process bind."""
    from plyngent.tools.context import get_session

    session = get_session()
    if session is not None and session.on_todo_change is not None:
        session.on_todo_change()
        return
    if _on_change is not None:
        _on_change()


def _process_stack() -> TodoStack:
    if _stack is None:
        msg = "todo stack is not available in this session"
        raise RuntimeError(msg)
    return _stack


async def _with_todo_stack(mutator: Callable[[TodoStack], str]) -> str:
    """Run *mutator* against the session todo stack and publish changes.

    Prefer::

        async with session.data:
            stack = session.data["todo"].typed(TodoStack)
            ...

    When a host already bound ``session.todo`` / process ``set_todo_stack``,
    mutate that live object and still refresh the view buffer when a session
    is bound so commits stay consistent.
    """
    from plyngent.agent.todo_stack import TodoStack
    from plyngent.tools.context import get_session

    session = get_session()
    if session is None:
        result = mutator(_process_stack())
        _notify()
        return result

    result = ""
    async with session.data as data:
        # Prefer host-bound live stack (CLI keeps TodoStack for nags / memory).
        if session.todo is not None:
            stack = session.todo
        elif _stack is not None:
            stack = _stack
            session.todo = stack
        else:
            stack = data["todo"].typed(TodoStack)
            session.todo = stack
        # Keep view domain + buffer in sync for durable commit.
        data["todo"].store(stack)
        result = mutator(stack)
    # View commit serialized to_raw; host on_todo_change may persist CLI memory.
    _notify()
    return result


@tool(name="todo_list", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_list() -> str:
    """Show the LIFO stack of **task groups** (TOP group = current breakdown level).

    Non-empty stack with open (pending/in_progress) items = unfinished work.
    All-terminal but non-empty = hygiene only (pop/clear). Push creates one
    group of siblings; pop removes the whole top group.
    Pattern: push [T1,T2] → push [T1.1,T1.2] → finish children → pop → push [T2.1]…
    """

    def _run(stack: TodoStack) -> str:
        stack.mark_touched()
        return stack.render()

    return await _with_todo_stack(_run)


@tool(name="todo_push", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_push(titles: list[str], notes: str = "") -> str:
    """Push **one task group** (siblings) onto the stack — not one level per title.

    ``titles``: JSON array of strings (tool schema type ``array``). All entries
    become members of a single new TOP group. Example: ``[\"T1\", \"T2\"]`` pushes
    one group {T1, T2}; a later ``[\"T1.1\", \"T1.2\"]`` pushes a child group above it.
    """

    def _run(stack: TodoStack) -> str:
        parsed = [t.strip() for t in titles if t and t.strip()]
        if not parsed:
            return "error: titles must be a non-empty array of strings"
        try:
            group = stack.push_group(parsed, notes=notes)
        except ValueError as exc:
            return f"error: {exc}"
        ids = ", ".join(i.id for i in group.items)
        return f"pushed group (depth={stack.depth}) items=[{ids}]\n{stack.render()}"

    return await _with_todo_stack(_run)


@tool(name="todo_pop", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_pop() -> str:
    """Pop the entire **top group** (all siblings from that push).

    Prefer after TOP items are done/cancelled so the stack does not stay
    non-empty with only finished work.
    """

    def _run(stack: TodoStack) -> str:
        group = stack.pop()
        if group is None:
            return "todo stack empty"
        titles = ", ".join(f"{i.id}:{i.title}" for i in group.items) or "(empty)"
        top = stack.top_group
        top_s = "(empty)" if top is None else ", ".join(i.id for i in top.items)
        return f"popped TOP group ({titles}); new top group=[{top_s}]\n{stack.render()}"

    return await _with_todo_stack(_run)


@tool(name="todo_update", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_update(
    item_id: str,
    status: str = "",
    title: str = "",
    notes: str = "",
) -> str:
    """Update a task by id inside any group.

    Open items are unfinished work: mark done/cancelled when the work is truly
    finished (not just deferred). Pop the TOP group when that breakdown level
    is complete so the stack does not linger as false open work.
    """

    def _run(stack: TodoStack) -> str:
        status_arg: TodoStatus | None = None
        if status.strip():
            token = status.strip().lower()
            if token not in _VALID_STATUS:
                return "error: status must be pending, in_progress, done, or cancelled"
            status_arg = cast("TodoStatus", token)
        try:
            item = stack.update(
                item_id.strip(),
                title=title if title.strip() else None,
                status=status_arg,
                notes=notes if notes != "" else None,
            )
        except (KeyError, ValueError) as exc:
            return f"error: {exc}"
        return f"updated {item.id} → {item.status}: {item.title}\n{stack.render()}"

    return await _with_todo_stack(_run)


@tool(name="todo_clear", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_clear() -> str:
    """Clear all groups on the stack."""

    def _run(stack: TodoStack) -> str:
        n = stack.clear()
        return f"cleared {n} item(s)"

    return await _with_todo_stack(_run)


TODO_TOOLS = [todo_list, todo_push, todo_pop, todo_update, todo_clear]
