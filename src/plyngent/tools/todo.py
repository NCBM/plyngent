from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.agent import tool
from plyngent.agent.todo_stack import parse_push_titles

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
    """List the nested todo stack (frames = breakdown levels).

    Pattern: push [T1,T2] → push [T1.1,T1.2] → finish children → pop → push [T2.1]…
    """
    stack = _require_stack()
    stack.mark_touched()
    _notify()
    return stack.render()


@tool(name="todo_push")
def todo_push(titles: str, notes: str = "") -> str:
    """Push a new breakdown frame with one or more sibling tasks.

    ``titles``: single title, newline-separated list, ``;``-separated, or JSON
    string array. Creates a **new nesting level** (does not append to the current
    frame). Example sequence: push ``T1\\nT2`` then push ``T1.1\\nT1.2``.
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
    ids = ", ".join(i.id for i in items)
    return f"pushed frame depth={stack.depth} items=[{ids}]\n{stack.render()}"


@tool(name="todo_pop")
def todo_pop() -> str:
    """Pop the top breakdown frame (leave the current nesting level).

    Use after finishing children of a task (e.g. after T1.1/T1.2, pop back to
    the T1/T2 frame). Does not delete sibling frames below.
    """
    stack = _require_stack()
    frame = stack.pop()
    if frame is None:
        return "todo stack empty"
    _notify()
    titles = ", ".join(f"{i.id}:{i.title}" for i in frame.items) or "(empty)"
    return f"popped frame ({len(frame.items)} item(s): {titles})\n{stack.render()}"


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
    """Clear the entire todo stack (all frames)."""
    stack = _require_stack()
    n = stack.clear()
    _notify()
    return f"cleared {n} item(s)"


TODO_TOOLS = [todo_list, todo_push, todo_pop, todo_update, todo_clear]
