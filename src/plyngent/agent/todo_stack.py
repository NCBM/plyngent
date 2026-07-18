from __future__ import annotations

from typing import Literal, cast

import msgspec
from msgspec import Struct, field

type TodoStatus = Literal["pending", "in_progress", "done", "cancelled"]

_OPEN: frozenset[str] = frozenset({"pending", "in_progress"})


class TodoItem(Struct, omit_defaults=True):
    """One entry on the todo stack (top of stack = next to work)."""

    id: str
    title: str
    status: TodoStatus = "pending"
    notes: str = ""


class TodoStackData(Struct, omit_defaults=True):
    """Serializable stack: ``items[0]`` is bottom, ``items[-1]`` is **top** (LIFO)."""

    items: list[TodoItem] = field(default_factory=list)
    next_id: int = 1


class TodoStack:
    """True LIFO stack of sub-tasks (not a queue, not a flat checklist).

    - **Top** = last element = next work item.
    - ``push`` / multi-push: new items go on the **top**. When pushing several
      titles at once, the **first** title becomes the top (worked first); the
      rest sit under it in order (DFS: children first, then later siblings).
    - ``pop``: remove **only the top** item.

    Breakdown pattern::

        push T1, T2          # top=T1, under=T2
        push T1.1, T1.2      # top=T1.1, then T1.2, then T1, then T2
        pop / done …         # clear T1.1, T1.2, then T1
        push T2.1            # top=T2.1 over T2
    """

    def __init__(self, data: TodoStackData | None = None) -> None:
        self._data: TodoStackData = data if data is not None else TodoStackData()
        self._touched_this_turn: bool = False

    @property
    def items(self) -> list[TodoItem]:
        """Bottom → top order (last is top)."""
        return self._data.items

    @property
    def top(self) -> TodoItem | None:
        if not self._data.items:
            return None
        return self._data.items[-1]

    @property
    def touched_this_turn(self) -> bool:
        return self._touched_this_turn

    def begin_turn(self) -> None:
        self._touched_this_turn = False

    def mark_touched(self) -> None:
        self._touched_this_turn = True

    def open_items(self) -> list[TodoItem]:
        return [item for item in self._data.items if item.status in _OPEN]

    def is_empty(self) -> bool:
        return not self._data.items

    def needs_review(self) -> bool:
        return bool(self.open_items()) and not self._touched_this_turn

    def to_data(self) -> TodoStackData:
        return self._data

    @classmethod
    def from_raw(cls, raw: object | None) -> TodoStack:
        if raw is None:
            return cls()
        # Nested-frame shape (brief intermediate design) → flatten bottom→top.
        if isinstance(raw, dict) and "frames" in raw:
            try:
                frames_raw = cast("dict[str, object]", raw).get("frames")
                next_raw = cast("dict[str, object]", raw).get("next_id", 1)
                next_id = max(1, int(next_raw) if isinstance(next_raw, int | str) else 1)
                items: list[TodoItem] = []
                if isinstance(frames_raw, list):
                    for frame in cast("list[object]", frames_raw):
                        if not isinstance(frame, dict):
                            continue
                        frame_items = msgspec.convert(frame.get("items"), type=list[TodoItem])
                        items.extend(frame_items)
                return cls(TodoStackData(items=items, next_id=next_id))
            except msgspec.ValidationError, TypeError, ValueError:
                return cls()
        try:
            data = msgspec.convert(raw, type=TodoStackData)
        except msgspec.ValidationError, TypeError, ValueError:
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
            return "(todo stack empty — top is empty)"
        lines: list[str] = ["(LIFO stack: TOP = next to work)"]
        # Show top first so the model sees work order.
        for offset, item in enumerate(reversed(self._data.items)):
            mark = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "done": "[x]",
                "cancelled": "[-]",
            }.get(item.status, "[?]")
            note = f" — {item.notes}" if item.notes else ""
            tag = " TOP" if offset == 0 else ""
            lines.append(f"{mark}{tag} {item.id}: {item.title}{note}")
        return "\n".join(lines)

    def review_prompt(self) -> str:
        return (
            "Todo stack review (internal control — not a human message).\n"
            "This is a LIFO stack (not a queue): TOP is the only pop target and "
            "the next sub-task to execute. Open work remains and you did not call "
            "any todo_* tools this turn.\n"
            "Breakdown: push children onto the top, finish/pop them, then the "
            "parent (or next sibling under it) becomes top again. "
            "push [T1,T2] then push [T1.1,T1.2] then pop finished tops, then "
            "push [T2.1]…\n\n"
            f"Current stack:\n{self.render()}"
        )

    def _alloc(self, title: str, *, notes: str, status: TodoStatus) -> TodoItem:
        item_id = f"t{self._data.next_id}"
        self._data.next_id += 1
        return TodoItem(id=item_id, title=title, status=status, notes=notes)

    def push_titles(
        self,
        titles: list[str],
        *,
        notes: str = "",
        status: TodoStatus = "pending",
    ) -> list[TodoItem]:
        """Push one or more tasks onto the top (LIFO).

        Titles are pushed so the **first** listed title becomes the new **top**
        (worked first). Example: ``push_titles(["T1","T2"])`` → top=T1, under=T2.
        """
        cleaned = [t.strip() for t in titles if t and t.strip()]
        if not cleaned:
            msg = "at least one non-empty title is required"
            raise ValueError(msg)
        note = notes.strip()
        # Push last→first so first title ends on top.
        created_rev: list[TodoItem] = []
        for title in reversed(cleaned):
            item = self._alloc(title, notes=note, status=status)
            self._data.items.append(item)
            created_rev.append(item)
        self.mark_touched()
        # Return in the same order as *titles* (first = top).
        return list(reversed(created_rev))

    def push(self, title: str, *, notes: str = "", status: TodoStatus = "pending") -> TodoItem:
        return self.push_titles([title], notes=notes, status=status)[0]

    def pop(self) -> TodoItem | None:
        """Remove and return the **top** item only (classic stack pop)."""
        if not self._data.items:
            return None
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


def parse_push_titles(raw: str) -> list[str]:
    """Parse multi-title push: JSON array, newlines, or ``;``-separated."""
    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data: object = msgspec.json.decode(text.encode())
        except msgspec.DecodeError, UnicodeEncodeError:
            data = None
        if isinstance(data, list):
            out = [item.strip() for item in cast("list[object]", data) if isinstance(item, str) and item.strip()]
            if out:
                return out
    parts: list[str] = []
    for line in text.replace(";", "\n").splitlines():
        token = line.strip()
        if token:
            parts.append(token)
    return parts
