from __future__ import annotations

from typing import Literal, cast

import msgspec
from msgspec import Struct, field

type TodoStatus = Literal["pending", "in_progress", "done", "cancelled"]

_OPEN: frozenset[str] = frozenset({"pending", "in_progress"})
_REVIEW_OPEN_TITLE_LIMIT = 8


class TodoItem(Struct, omit_defaults=True):
    """One task inside a group (siblings share a group; not stacked individually)."""

    id: str
    title: str
    status: TodoStatus = "pending"
    notes: str = ""


class TodoGroup(Struct, omit_defaults=True):
    """One stack entry: a group of sibling tasks pushed together."""

    items: list[TodoItem] = field(default_factory=list)


class TodoStackData(Struct, omit_defaults=True):
    """LIFO of **groups**: ``groups[0]`` bottom, ``groups[-1]`` **TOP** group."""

    groups: list[TodoGroup] = field(default_factory=list)
    next_id: int = 1


class TodoStack:
    """LIFO stack of **task groups** (not a queue of individual tasks).

    - **Push** always creates **one new group** containing one or more sibling
      tasks (``push T1, T2`` is one frame, not two stack levels).
    - **Pop** removes the entire **top group**.
    - Within a group, update tasks by id (done / in_progress / …).

    Breakdown pattern::

        push [T1, T2]        # top group = {T1, T2}
        push [T1.1, T1.2]    # top group = {T1.1, T1.2}; under = {T1, T2}
        # finish T1.1 / T1.2 via update…
        pop                  # leave child group; top again = {T1, T2}
        push [T2.1]          # top group = {T2.1}
    """

    def __init__(self, data: TodoStackData | None = None) -> None:
        self._data: TodoStackData = data if data is not None else TodoStackData()
        self._touched_this_turn: bool = False

    @property
    def groups(self) -> list[TodoGroup]:
        """Bottom → top (last is top group)."""
        return self._data.groups

    @property
    def top_group(self) -> TodoGroup | None:
        if not self._data.groups:
            return None
        return self._data.groups[-1]

    @property
    def depth(self) -> int:
        return len(self._data.groups)

    @property
    def touched_this_turn(self) -> bool:
        return self._touched_this_turn

    def begin_turn(self) -> None:
        self._touched_this_turn = False

    def mark_touched(self) -> None:
        self._touched_this_turn = True

    def all_items(self) -> list[TodoItem]:
        return [item for group in self._data.groups for item in group.items]

    def open_items(self) -> list[TodoItem]:
        return [item for item in self.all_items() if item.status in _OPEN]

    def is_empty(self) -> bool:
        return not self._data.groups

    def needs_review(self) -> bool:
        """True when end-of-turn should inject a todo nag.

        Only when the stack is non-empty **and** no ``todo_*`` tool ran this
        turn (list/push/pop/update/clear all call :meth:`mark_touched`).

        Open items alone must not re-nag after the model already accessed the
        stack — that caused improper ``synthetic_tool`` injects mid-flow.
        Hygiene (all-terminal non-empty) and open work share the same gate:
        ignore todos for the whole turn → nag once; touch once → no end nag.
        """
        if self.is_empty():
            return False
        return not self._touched_this_turn

    def to_data(self) -> TodoStackData:
        return self._data

    @classmethod
    def from_raw(cls, raw: object | None) -> TodoStack:  # noqa: C901, PLR0911
        if raw is None or not isinstance(raw, dict):
            return cls()
        blob = cast("dict[str, object]", raw)

        # Current shape: {groups: [...], next_id}
        if "groups" in blob:
            try:
                data = msgspec.convert(raw, type=TodoStackData)
            except msgspec.ValidationError, TypeError, ValueError:
                return cls()
            if data.next_id < 1:
                data = msgspec.structs.replace(data, next_id=1)
            return cls(data)

        # Intermediate nested frames → groups (same structure).
        if "frames" in blob:
            try:
                frames_raw = blob.get("frames")
                next_raw = blob.get("next_id", 1)
                next_id = max(1, int(next_raw) if isinstance(next_raw, int | str) else 1)
                groups: list[TodoGroup] = []
                if isinstance(frames_raw, list):
                    for frame in cast("list[object]", frames_raw):
                        if not isinstance(frame, dict):
                            continue
                        frame_map = cast("dict[str, object]", frame)
                        frame_items = msgspec.convert(frame_map.get("items"), type=list[TodoItem])
                        if frame_items:
                            groups.append(TodoGroup(items=frame_items))
                return cls(TodoStackData(groups=groups, next_id=next_id))
            except msgspec.ValidationError, TypeError, ValueError:
                return cls()

        # Flat items list → one group (legacy).
        if "items" in blob:
            try:
                items = msgspec.convert(blob.get("items"), type=list[TodoItem])
                next_raw = blob.get("next_id", 1)
                next_id = max(1, int(next_raw) if isinstance(next_raw, int | str) else 1)
            except msgspec.ValidationError, TypeError, ValueError:
                return cls()
            groups = [TodoGroup(items=items)] if items else []
            return cls(TodoStackData(groups=groups, next_id=next_id))

        return cls()

    def to_raw(self) -> dict[str, object]:
        out: object = msgspec.to_builtins(self._data)
        if not isinstance(out, dict):
            return {"groups": [], "next_id": 1}
        raw = cast("dict[object, object]", out)
        return {str(key): value for key, value in raw.items()}

    def render(self) -> str:
        if not self._data.groups:
            return "(todo stack empty — no groups)"
        lines: list[str] = [
            f"(LIFO of groups: depth={self.depth}; TOP group = current breakdown level)",
        ]
        # Top group first for the model.
        for offset, group in enumerate(reversed(self._data.groups)):
            depth = self.depth - 1 - offset
            tag = " TOP" if offset == 0 else ""
            lines.append(f"group d={depth}{tag}:")
            if not group.items:
                lines.append("  (empty group)")
                continue
            for item in group.items:
                mark = {
                    "pending": "[ ]",
                    "in_progress": "[~]",
                    "done": "[x]",
                    "cancelled": "[-]",
                }.get(item.status, "[?]")
                note = f" — {item.notes}" if item.notes else ""
                lines.append(f"  {mark} {item.id}: {item.title}{note}")
        return "\n".join(lines)

    def turn_reminder_prompt(self) -> str:
        """Mid-context nudge when a turn starts with a non-empty stack.

        A non-empty stack usually means unfinished work (open items) or unfinished
        stack hygiene (terminal items still grouped). Prefer finishing real work
        over pure bookkeeping when both exist.
        """
        n_open = len(self.open_items())
        n_groups = self.depth
        if n_open:
            headline = (
                f"[TODO OPEN WORK] Stack not empty: {n_open} open item(s) across "
                f"{n_groups} group(s). This is unfinished work from earlier in the "
                "session — not optional decoration."
            )
            action = (
                "Treat open items as active commitments: continue them, mark "
                "done/cancelled when finished, or push a child breakdown for the "
                "current TOP item. Only todo_clear if the user abandoned the plan. "
                "Do not ignore the stack while answering unrelated chatter."
            )
        else:
            headline = (
                f"[TODO HYGIENE] Stack not empty: {n_groups} group(s) with no open "
                "items (all done/cancelled). Work may be finished; the stack is not."
            )
            action = (
                "Pop finished TOP groups (or todo_clear when the whole plan is done). "
                "Leaving only-terminal groups wastes context; clean up when you can."
            )
        lines = [
            headline,
            action,
            "Tools: todo_list | todo_push(titles=[...]) | todo_update | todo_pop | todo_clear",
            "Rules: TOP = current level; push = one sibling group; pop = whole TOP group.",
            "Stack:",
            self.render(),
        ]
        return "\n".join(lines)

    def review_prompt(self) -> str:
        """End-of-turn nag: non-empty stack still signals unfinished work."""
        open_items = self.open_items()
        n_open = len(open_items)
        n_groups = self.depth
        if n_open:
            open_titles = "; ".join(f"{item.id}:{item.title}" for item in open_items[:_REVIEW_OPEN_TITLE_LIMIT])
            if n_open > _REVIEW_OPEN_TITLE_LIMIT:
                open_titles += f"; …(+{n_open - _REVIEW_OPEN_TITLE_LIMIT} more)"
            headline = (
                f"[TODO OPEN WORK] You stopped with {n_open} open item(s) still on "
                f"the stack ({n_groups} group(s)). That usually means undone work."
            )
            action = (
                "Do not end the turn while open tasks remain unaddressed. Next: "
                "resume an open item, mark done/cancelled, pop a finished TOP group, "
                "or push a child breakdown. Clear only if the user no longer wants "
                f"the plan. Open: {open_titles}"
            )
        else:
            headline = (
                f"[TODO HYGIENE] Stack still has {n_groups} group(s) but every item "
                "is done/cancelled. Real work looks finished; stack cleanup does not."
            )
            action = (
                "Pop finished TOP groups (or todo_clear when the whole plan is done). "
                "A non-empty all-terminal stack is unfinished task hygiene, not new work."
            )
        lines = [
            headline,
            action,
            "Tools: todo_list | todo_push(titles=[...]) | todo_update | todo_pop | todo_clear",
            "Rules: TOP group = current level; push=one sibling group; pop=remove whole TOP group.",
            "Stack:",
            self.render(),
        ]
        return "\n".join(lines)

    def _existing_numeric_ids(self) -> list[int]:
        """Numeric suffixes of ids that look like ``tN`` (for counter reuse)."""
        return [
            int(item.id[1:])
            for group in self._data.groups
            for item in group.items
            if item.id.startswith("t") and item.id[1:].isdigit()
        ]

    def _sync_next_id(self) -> None:
        """Set ``next_id`` to one past the highest live id (or 1 if empty)."""
        nums = self._existing_numeric_ids()
        self._data.next_id = max(nums) + 1 if nums else 1

    def _alloc(self, title: str, *, notes: str, status: TodoStatus) -> TodoItem:
        item_id = f"t{self._data.next_id}"
        self._data.next_id += 1
        return TodoItem(id=item_id, title=title, status=status, notes=notes)

    def push_group(
        self,
        titles: list[str],
        *,
        notes: str = "",
        status: TodoStatus = "pending",
    ) -> TodoGroup:
        """Push **one** new group containing all *titles* as siblings."""
        cleaned = [t.strip() for t in titles if t and t.strip()]
        if not cleaned:
            msg = "at least one non-empty title is required"
            raise ValueError(msg)
        note = notes.strip()
        # Rebase counter on live ids so clear/pop do not leave a high watermark.
        self._sync_next_id()
        items = [self._alloc(title, notes=note, status=status) for title in cleaned]
        group = TodoGroup(items=items)
        self._data.groups.append(group)
        self.mark_touched()
        return group

    def push(self, title: str, *, notes: str = "", status: TodoStatus = "pending") -> TodoItem:
        """Push a group with a single task (still one stack level)."""
        group = self.push_group([title], notes=notes, status=status)
        return group.items[0]

    def pop(self) -> TodoGroup | None:
        """Pop and return the **top group** (all siblings in that push)."""
        if not self._data.groups:
            return None
        group = self._data.groups.pop()
        self._sync_next_id()
        self.mark_touched()
        return group

    def clear(self) -> int:
        n = sum(len(g.items) for g in self._data.groups)
        self._data.groups.clear()
        self._sync_next_id()
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
        for group in self._data.groups:
            for index, item in enumerate(group.items):
                if item.id != item_id:
                    continue
                new_title = title.strip() if title is not None else item.title
                if not new_title:
                    msg = "title must not be empty"
                    raise ValueError(msg)
                new_status = status if status is not None else item.status
                new_notes = notes if notes is not None else item.notes
                updated = TodoItem(id=item.id, title=new_title, status=new_status, notes=new_notes)
                group.items[index] = updated
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
