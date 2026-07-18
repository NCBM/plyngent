from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.agent import tool
from plyngent.agent.todo_stack import parse_push_titles

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
    if _stack is None:
        msg = "todo stack is not available in this session"
        raise RuntimeError(msg)
    return _stack


def _notify() -> None:
    if _on_change is not None:
        _on_change()


@tool(name="todo_list")
def todo_list() -> str:
    """Show the LIFO todo stack (TOP = next sub-task to work / pop).

    Not a queue: only the top is popped. Breakdown: push children on top of a
    parent, finish and pop them, then the parent (or next sibling) is top again.
    """
    stack = _require_stack()
    stack.mark_touched()
    _notify()
    return stack.render()


@tool(name="todo_push")
def todo_push(titles: str, notes: str = "") -> str:
    """Push task(s) onto the **top** of the LIFO stack.

    ``titles``: one title, newlines, ``;``, or JSON array. First title becomes
    the new TOP (worked first). Example: ``T1\\nT2`` → top=T1, under=T2; then
    ``T1.1\\nT1.2`` → top=T1.1 over T1.2 over T1 over T2.
    """
    stack = _require_stack()
    parsed = parse_push_titles(titles)
    if not parsed:
        return "error: titles must contain at least one non-empty title"
    try:
        items = stack.push_titles(parsed, notes=notes)
    except ValueError as exc:
        return f"error: {exc}"
    _notify()
    top = stack.top
    top_s = f"{top.id}:{top.title}" if top else "?"
    ids = ", ".join(i.id for i in items)
    return f"pushed [{ids}] (top now {top_s})\n{stack.render()}"


@tool(name="todo_pop")
def todo_pop() -> str:
    """Pop the **top** item only (classic stack). Does not remove items under it."""
    stack = _require_stack()
    item = stack.pop()
    if item is None:
        return "todo stack empty"
    _notify()
    top = stack.top
    top_s = f"{top.id}:{top.title}" if top else "(empty)"
    return f"popped TOP {item.id}: {item.title} ({item.status}); new top={top_s}\n{stack.render()}"


@tool(name="todo_update")
def todo_update(
    item_id: str,
    status: str = "",
    title: str = "",
    notes: str = "",
) -> str:
    """Update a todo by id (any depth). Prefer pop when the top task is finished."""
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


@tool(name="todo_clear")
def todo_clear() -> str:
    """Clear the entire stack."""
    stack = _require_stack()
    n = stack.clear()
    _notify()
    return f"cleared {n} item(s)"


TODO_TOOLS = [todo_list, todo_push, todo_pop, todo_update, todo_clear]
