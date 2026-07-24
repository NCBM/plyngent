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
    """Bind the session todo stack for tool handlers (and optional persist hook)."""
    global _stack, _on_change  # noqa: PLW0603 — intentional process bind
    _stack = stack
    _on_change = on_change


def get_todo_stack() -> TodoStack | None:
    return _stack


def _require_stack() -> TodoStack:
    """Prefer session-bound todo; fall back to process bind during migration.

    Order: ``session.todo`` facet → ``session.data["todo"].typed(TodoStack)`` when a
    view is open/bound → process ``set_todo_stack`` global.
    """
    from plyngent.agent.todo_stack import TodoStack
    from plyngent.tools.context import get_session

    session = get_session()
    if session is not None:
        if session.todo is not None:
            return session.todo
        try:
            # Prefer live domain object when a txn is open; otherwise skip.
            todo = session.data["todo"].typed(TodoStack)
        except RuntimeError:
            todo = None
        if isinstance(todo, TodoStack):
            session.todo = todo
            return todo
    if _stack is None:
        msg = "todo stack is not available in this session"
        raise RuntimeError(msg)
    return _stack


def _notify() -> None:
    if _on_change is not None:
        _on_change()
    # Session-bound stacks may also persist via host on_change on the process bind.


@tool(name="todo_list", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_list() -> str:
    """Show the LIFO stack of **task groups** (TOP group = current breakdown level).

    Non-empty stack with open (pending/in_progress) items = unfinished work.
    All-terminal but non-empty = hygiene only (pop/clear). Push creates one
    group of siblings; pop removes the whole top group.
    Pattern: push [T1,T2] → push [T1.1,T1.2] → finish children → pop → push [T2.1]…
    """
    stack = _require_stack()
    stack.mark_touched()
    _notify()
    return stack.render()


@tool(name="todo_push", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_push(titles: list[str], notes: str = "") -> str:
    """Push **one task group** (siblings) onto the stack — not one level per title.

    ``titles``: JSON array of strings (tool schema type ``array``). All entries
    become members of a single new TOP group. Example: ``[\"T1\", \"T2\"]`` pushes
    one group {T1, T2}; a later ``[\"T1.1\", \"T1.2\"]`` pushes a child group above it.
    """
    stack = _require_stack()
    parsed = [t.strip() for t in titles if t and t.strip()]
    if not parsed:
        return "error: titles must be a non-empty array of strings"
    try:
        group = stack.push_group(parsed, notes=notes)
    except ValueError as exc:
        return f"error: {exc}"
    _notify()
    ids = ", ".join(i.id for i in group.items)
    return f"pushed group (depth={stack.depth}) items=[{ids}]\n{stack.render()}"


@tool(name="todo_pop", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_pop() -> str:
    """Pop the entire **top group** (all siblings from that push).

    Prefer after TOP items are done/cancelled so the stack does not stay
    non-empty with only finished work.
    """
    stack = _require_stack()
    group = stack.pop()
    if group is None:
        return "todo stack empty"
    _notify()
    titles = ", ".join(f"{i.id}:{i.title}" for i in group.items) or "(empty)"
    top = stack.top_group
    top_s = "(empty)" if top is None else ", ".join(i.id for i in top.items)
    return f"popped TOP group ({titles}); new top group=[{top_s}]\n{stack.render()}"


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
    stack = _require_stack()
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
    _notify()
    return f"updated {item.id} → {item.status}: {item.title}\n{stack.render()}"


@tool(name="todo_clear", tags=ToolTag.LOCAL | ToolTag.PUBLIC | ToolTag.SESSION_STATE)
async def todo_clear() -> str:
    """Clear all groups on the stack."""
    stack = _require_stack()
    n = stack.clear()
    _notify()
    return f"cleared {n} item(s)"


TODO_TOOLS = [todo_list, todo_push, todo_pop, todo_update, todo_clear]
