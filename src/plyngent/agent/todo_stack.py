from __future__ import annotations

from typing import Literal, cast

import msgspec
from msgspec import Struct, field

type TodoStatus = Literal["pending", "in_progress", "done", "cancelled"]

_OPEN: frozenset[str] = frozenset({"pending", "in_progress"})


class TodoItem(Struct, omit_defaults=True):
    """One sub-task on the session todo stack."""

    id: str
    title: str
    status: TodoStatus = "pending"
    notes: str = ""


class TodoStackData(Struct, omit_defaults=True):
    """Serializable stack body (session storage)."""

    items: list[TodoItem] = field(default_factory=list)
    next_id: int = 1


class TodoStack:
    """Session-local ordered sub-tasks for breaking work into steps.

    Maintained mainly by model tools; humans can show/push/pop/clear via slash.
    """

    def __init__(self, data: TodoStackData | None = None) -> None:
        self._data: TodoStackData = data if data is not None else TodoStackData()
        self._touched_this_turn: bool = False

    @property
    def items(self) -> list[TodoItem]:
        return self._data.items

    @property
    def touched_this_turn(self) -> bool:
        return self._touched_this_turn

    def begin_turn(self) -> None:
        """Reset the per-turn touch flag (call at start of each user turn)."""
        self._touched_this_turn = False

    def mark_touched(self) -> None:
        self._touched_this_turn = True

    def open_items(self) -> list[TodoItem]:
        return [item for item in self._data.items if item.status in _OPEN]

    def is_empty(self) -> bool:
        return not self._data.items

    def needs_review(self) -> bool:
        """True when open work exists but no todo tool ran this turn."""
        return bool(self.open_items()) and not self._touched_this_turn

    def to_data(self) -> TodoStackData:
        return self._data

    @classmethod
    def from_raw(cls, raw: object | None) -> TodoStack:
        if raw is None:
            return cls()
        try:
            data = msgspec.convert(raw, type=TodoStackData)
        except (msgspec.ValidationError, TypeError, ValueError):
            return cls()
        if data.next_id < 1:
            data = msgspec.structs.replace(data, next_id=1)
        return cls(data)

    def to_raw(self) -> dict[str, object]:
        out: object = msgspec.to_builtins(self._data)
        if not isinstance(out, dict):
            return {"items": [], "next_id": 1}
        raw = cast("dict[object, object]", out)
        return {str(key): value for key, value in raw.items()}

    def render(self) -> str:
        if not self._data.items:
            return "(todo stack empty)"
        lines: list[str] = []
        for item in self._data.items:
            mark = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "done": "[x]",
                "cancelled": "[-]",
            }.get(item.status, "[?]")
            note = f" — {item.notes}" if item.notes else ""
            lines.append(f"{mark} {item.id}: {item.title}{note}")
        return "\n".join(lines)

    def review_prompt(self) -> str:
        """Control-message body (developer role) when the model finishes without todo ops."""
        return (
            "Todo stack review (internal control — not a human message).\n"
            "Open sub-tasks remain and you did not call any todo_* tools this turn.\n"
            "Review the stack: mark finished items done, update in_progress, "
            "push new sub-tasks, or pop/cancel obsolete ones. Then continue the user work.\n\n"
            f"Current stack:\n{self.render()}"
        )

    def push(self, title: str, *, notes: str = "", status: TodoStatus = "pending") -> TodoItem:
        token = title.strip()
        if not token:
            msg = "title must not be empty"
            raise ValueError(msg)
        item_id = f"t{self._data.next_id}"
        self._data.next_id += 1
        item = TodoItem(id=item_id, title=token, status=status, notes=notes.strip())
        self._data.items.append(item)
        self.mark_touched()
        return item

    def pop(self) -> TodoItem | None:
        """Remove and return the last open item, else the last item, else None."""
        if not self._data.items:
            return None
        open_ids = {i.id for i in self.open_items()}
        for index in range(len(self._data.items) - 1, -1, -1):
            if self._data.items[index].id in open_ids:
                item = self._data.items.pop(index)
                self.mark_touched()
                return item
        item = self._data.items.pop()
        self.mark_touched()
        return item

    def clear(self) -> int:
        n = len(self._data.items)
        self._data.items.clear()
        if n:
            self.mark_touched()
        return n

    def get(self, item_id: str) -> TodoItem | None:
        for item in self._data.items:
            if item.id == item_id:
                return item
        return None

    def update(
        self,
        item_id: str,
        *,
        title: str | None = None,
        status: TodoStatus | None = None,
        notes: str | None = None,
    ) -> TodoItem:
        for index, item in enumerate(self._data.items):
            if item.id != item_id:
                continue
            new_title = title.strip() if title is not None else item.title
            if not new_title:
                msg = "title must not be empty"
                raise ValueError(msg)
            new_status = status if status is not None else item.status
            new_notes = notes if notes is not None else item.notes
            updated = TodoItem(id=item.id, title=new_title, status=new_status, notes=new_notes)
            self._data.items[index] = updated
            self.mark_touched()
            return updated
        msg = f"unknown todo id {item_id!r}"
        raise KeyError(msg)
