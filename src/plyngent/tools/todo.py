from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.agent import tool

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.agent.todo_stack import TodoStack, TodoStatus

# Module-level bind (same pattern as workspace root) for @tool handlers.
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
    """List the current todo/task stack (sub-tasks for multi-step work)."""
    stack = _require_stack()
    stack.mark_touched()
    _notify()
    return stack.render()


@tool(name="todo_push")
def todo_push(title: str, notes: str = "") -> str:
    """Push a new sub-task onto the todo stack (pending)."""
    stack = _require_stack()
    try:
        item = stack.push(title, notes=notes)
    except ValueError as exc:
        return f"error: {exc}"
    _notify()
    return f"pushed {item.id}: {item.title}\n{stack.render()}"


@tool(name="todo_pop")
def todo_pop() -> str:
    """Pop the last open sub-task (or last item if all closed)."""
    stack = _require_stack()
    item = stack.pop()
    if item is None:
        return "todo stack empty"
    _notify()
    return f"popped {item.id}: {item.title} ({item.status})\n{stack.render()}"


@tool(name="todo_update")
def todo_update(
    item_id: str,
    status: str = "",
    title: str = "",
    notes: str = "",
) -> str:
    """Update a todo by id. ``status``: pending|in_progress|done|cancelled."""
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
    """Clear the entire todo stack."""
    stack = _require_stack()
    n = stack.clear()
    _notify()
    return f"cleared {n} item(s)"


TODO_TOOLS = [todo_list, todo_push, todo_pop, todo_update, todo_clear]
