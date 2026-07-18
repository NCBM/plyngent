from __future__ import annotations

from typing import Literal, cast

import msgspec
from msgspec import Struct, field

type TodoStatus = Literal["pending", "in_progress", "done", "cancelled"]

_OPEN: frozenset[str] = frozenset({"pending", "in_progress"})


class TodoItem(Struct, omit_defaults=True):
    """One sub-task within a stack frame."""

    id: str
    title: str
    status: TodoStatus = "pending"
    notes: str = ""


class TodoFrame(Struct, omit_defaults=True):
    """One nesting level: sibling tasks pushed together."""

    items: list[TodoItem] = field(default_factory=list)


class TodoStackData(Struct, omit_defaults=True):
    """Serializable stack body (session storage).

    ``frames`` is bottom→top: root plan first, deepest breakdown last.
    Pattern: push [T1,T2] → push [T1.1,T1.2] → pop → push [T2.1] → …
    """

    frames: list[TodoFrame] = field(default_factory=list)
    next_id: int = 1


class TodoStack:
    """Nested todo stack for hierarchical sub-task breakdown.

    Each ``push`` of one or more titles creates a **new frame** (deeper level).
    ``pop`` removes the **top frame** (leave a breakdown level).
    Model tools maintain the stack; humans use ``/todos``.
    """

    def __init__(self, data: TodoStackData | None = None) -> None:
        self._data: TodoStackData = data if data is not None else TodoStackData()
        self._touched_this_turn: bool = False

    @property
    def frames(self) -> list[TodoFrame]:
        return self._data.frames

    @property
    def depth(self) -> int:
        return len(self._data.frames)

    @property
    def touched_this_turn(self) -> bool:
        return self._touched_this_turn

    def begin_turn(self) -> None:
        """Reset the per-turn touch flag (call at start of each user turn)."""
        self._touched_this_turn = False

    def mark_touched(self) -> None:
        self._touched_this_turn = True

    def all_items(self) -> list[TodoItem]:
        return [item for frame in self._data.frames for item in frame.items]

    def open_items(self) -> list[TodoItem]:
        return [item for item in self.all_items() if item.status in _OPEN]

    def is_empty(self) -> bool:
        return not self._data.frames

    def needs_review(self) -> bool:
        """True when open work exists but no todo tool ran this turn."""
        return bool(self.open_items()) and not self._touched_this_turn

    def to_data(self) -> TodoStackData:
        return self._data

    @classmethod
    def from_raw(cls, raw: object | None) -> TodoStack:
        if raw is None:
            return cls()
        # Legacy flat shape: {items: [...], next_id: N} → single root frame.
        if isinstance(raw, dict) and "frames" not in raw and "items" in raw:
            try:
                flat = cast("dict[str, object]", raw)
                items = msgspec.convert(flat.get("items"), type=list[TodoItem])
                next_raw = flat.get("next_id", 1)
                next_id = max(1, int(next_raw) if isinstance(next_raw, int | str) else 1)
            except (msgspec.ValidationError, TypeError, ValueError):
                return cls()
            data = TodoStackData(
                frames=[TodoFrame(items=items)] if items else [],
                next_id=next_id,
            )
            return cls(data)
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
            return {"frames": [], "next_id": 1}
        raw = cast("dict[object, object]", out)
        return {str(key): value for key, value in raw.items()}

    def render(self) -> str:
        if not self._data.frames:
            return "(todo stack empty)"
        lines: list[str] = [f"(depth={self.depth})"]
        for depth, frame in enumerate(self._data.frames):
            indent = "  " * depth
            lines.append(f"{indent}frame {depth}:")
            if not frame.items:
                lines.append(f"{indent}  (empty frame)")
                continue
            for item in frame.items:
                mark = {
                    "pending": "[ ]",
                    "in_progress": "[~]",
                    "done": "[x]",
                    "cancelled": "[-]",
                }.get(item.status, "[?]")
                note = f" — {item.notes}" if item.notes else ""
                lines.append(f"{indent}  {mark} {item.id}: {item.title}{note}")
        return "\n".join(lines)

    def review_prompt(self) -> str:
        """Control-message body (developer role) when the model finishes without todo ops."""
        return (
            "Todo stack review (internal control — not a human message).\n"
            "Open sub-tasks remain and you did not call any todo_* tools this turn.\n"
            "Use nested push/pop for breakdown: push [T1,T2], push [T1.1,T1.2], "
            "finish children, pop frame, then push children of T2, etc. "
            "Mark done/cancelled, push new frames, or pop a finished level.\n\n"
            f"Current stack:\n{self.render()}"
        )

    def push_titles(
        self,
        titles: list[str],
        *,
        notes: str = "",
        status: TodoStatus = "pending",
    ) -> list[TodoItem]:
        """Push a new frame containing *titles* (one or more siblings at this depth)."""
        cleaned = [t.strip() for t in titles if t and t.strip()]
        if not cleaned:
            msg = "at least one non-empty title is required"
            raise ValueError(msg)
        note = notes.strip()
        created: list[TodoItem] = []
        for title in cleaned:
            item_id = f"t{self._data.next_id}"
            self._data.next_id += 1
            created.append(TodoItem(id=item_id, title=title, status=status, notes=note))
        self._data.frames.append(TodoFrame(items=created))
        self.mark_touched()
        return created

    def push(self, title: str, *, notes: str = "", status: TodoStatus = "pending") -> TodoItem:
        """Push a new frame with a single task (convenience for one title)."""
        items = self.push_titles([title], notes=notes, status=status)
        return items[0]

    def pop(self) -> TodoFrame | None:
        """Pop and return the top frame (leave the current breakdown level)."""
        if not self._data.frames:
            return None
        frame = self._data.frames.pop()
        self.mark_touched()
        return frame

    def clear(self) -> int:
        n = sum(len(frame.items) for frame in self._data.frames)
        self._data.frames.clear()
        if n:
            self.mark_touched()
        return n

    def get(self, item_id: str) -> TodoItem | None:
        for item in self.all_items():
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
        for frame in self._data.frames:
            for index, item in enumerate(frame.items):
                if item.id != item_id:
                    continue
                new_title = title.strip() if title is not None else item.title
                if not new_title:
                    msg = "title must not be empty"
                    raise ValueError(msg)
                new_status = status if status is not None else item.status
                new_notes = notes if notes is not None else item.notes
                updated = TodoItem(id=item.id, title=new_title, status=new_status, notes=new_notes)
                frame.items[index] = updated
                self.mark_touched()
                return updated
        msg = f"unknown todo id {item_id!r}"
        raise KeyError(msg)


def parse_push_titles(raw: str) -> list[str]:
    """Parse multi-title push input: JSON array, or newline/semicolon-separated."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data: object = msgspec.json.decode(text.encode())
        except (msgspec.DecodeError, UnicodeEncodeError):
            data = None
        if isinstance(data, list):
            out = [item.strip() for item in cast("list[object]", data) if isinstance(item, str) and item.strip()]
            if out:
                return out
    # Newlines, then ``;`` as separators.
    parts: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        token = line.strip()
        if token:
            parts.append(token)
    return parts
