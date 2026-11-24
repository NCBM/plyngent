"""The todo stack, as a value: groups of sibling tasks, bottom to top.

A conversation's plan is a LIFO of **groups** and not a list of tasks. One push is one level of
breakdown — "write the parser" and "test it" pushed together are two siblings at one level, not two
levels — and a pop takes that whole level back off, so what is left is the question the level was
answering. An individual task is finished through :meth:`Todos.update`, never by being taken out from
under the ones beside it.

Two numbers address a task, and neither moves when something is pushed above it:

* a group's **depth** — ``0`` is the bottom, the last group is the TOP one — so pushing a child
  breakdown does not rename the parent it came from;
* an item's **number inside its group**, counting from 1 in the order the listing prints.

That is a deliberate difference from ``main``, which numbers every item of the stack in one sequence,
TOP group first: one number is easier to say, but every push and pop renumbers everything, so nothing
can be talked about twice without listing it again. Here the price is a second number and the gain is
that "group 0, item 1" identifies the same task until it is changed.

Nothing here knows about a session, a journal or a clock: the stack is read and written as JSON text
by the session that owns it (§10.9), so this module is about the shape alone.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import msgspec
from msgspec import Struct, field

from plyngent.kernel.words import counted

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable

#: What a listing says when the stack holds nothing, which is a state and not a missing answer.
EMPTY = "(todo stack empty — no groups)"

#: How many open tasks a reminder names by title before it counts the rest: enough to work from, short
#: enough that a line written once does not sit in every later request at full length.
NAMED = 3


class TodoStatus(StrEnum):
    """Where one task stands.

    The members are declared in name order and not in the order a task moves through them, because
    this is the closed set a tool's schema carries and a schema is written in a stable order (§10.5).
    Which of them still count as work is :data:`OPEN`.

    ``PAUSED`` is the one that is neither work nor finished: something that cannot be done now, kept in
    the plan so the person can see it, and out of ``OPEN`` so the plan can be at rest while it waits
    (E188).
    """

    CANCELLED = "cancelled"
    DONE = "done"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    PENDING = "pending"


#: The statuses that are still work: an item is open until somebody finishes it or gives it up.
OPEN: frozenset[TodoStatus] = frozenset({TodoStatus.PENDING, TodoStatus.IN_PROGRESS})

#: What a listing writes in front of an item, by status — the marks a checklist uses.
MARKS: dict[TodoStatus, str] = {
    TodoStatus.CANCELLED: "[-]",
    TodoStatus.DONE: "[x]",
    TodoStatus.IN_PROGRESS: "[~]",
    TodoStatus.PAUSED: "[!]",
    TodoStatus.PENDING: "[ ]",
}


class TodoItem(Struct, kw_only=True):
    """One task: what it is, where it stands, and what was noted about it.

    ``id`` is the stored key and nothing else — a listing numbers items by their place in their
    group, so the id never has to move when the order a task was written in stops being the order it
    is read in. It is kept because two items may carry the same title, and something has to tell them
    apart in the text this is written down as.
    """

    id: int
    title: str
    status: TodoStatus = TodoStatus.PENDING
    notes: str = ""


class TodoGroup(Struct, kw_only=True):
    """One level of breakdown: the tasks that were pushed together.

    A group is not a tree node with a parent pointer. The stack *is* the nesting: the group above
    this one is why it exists, and nothing has to be kept in step when one of them goes away.
    """

    items: list[TodoItem] = field(default_factory=list)


class Todos(Struct, kw_only=True):
    """The stack: ``groups[0]`` is the bottom, ``groups[-1]`` is the TOP group.

    Every method that changes the stack changes this value in place, and the caller decides what to
    write down — which is what lets the session write one whole setting per change, with no difference
    for a reader to apply in order.
    """

    groups: list[TodoGroup] = field(default_factory=list)
    #: The id the next item gets. Stored with the stack so that a restart cannot hand out an id twice.
    next_id: int = 1

    def levels(self) -> int:
        """How many groups the stack holds — 0 for an empty stack.

        A level is one group, and the word is this method's on purpose: ``depth`` is a group's own number
        (``0`` is the bottom), so counting the groups with it would make one word answer "where is this
        group" and "how many are there" (E187).
        """
        return len(self.groups)

    def top_depth(self) -> int:
        """Which number the TOP group has — ``-1`` when the stack holds nothing.

        The one number a plan that drives itself is judged by: how deep the work has been broken down is
        where a person's floor is compared against, and an empty stack has no group to compare (E187).
        """
        return len(self.groups) - 1

    def group(self, depth: int) -> TodoGroup | None:
        """The group a listing wrote as number *depth* (0 is the bottom), or ``None``."""
        return self.groups[depth] if 0 <= depth < len(self.groups) else None

    def items(self) -> list[TodoItem]:
        """Every item on the stack, bottom group first."""
        return [item for group in self.groups for item in group.items]

    def open_items(self) -> list[TodoItem]:
        """Every item that is still work."""
        return [item for item in self.items() if item.status in OPEN]

    def paused_items(self) -> list[TodoItem]:
        """Every item that was set aside for now: still in the plan, and not work to do."""
        return [item for item in self.items() if item.status is TodoStatus.PAUSED]

    def has_work(self) -> bool:
        """Whether the plan still owes work — which is a question about ``OPEN`` and nothing else.

        The predicate and the sentence are two things (:meth:`reminder` is the sentence), because a plan
        whose every item is paused has nothing to do and something to say: reading "is there work" off a
        line of prose is how a plan that is waiting comes to be read as a plan that is finished (E188).
        """
        return any(item.status in OPEN for item in self.items())

    def push_group(self, titles: Iterable[str], *, notes: str = "") -> TodoGroup:
        """Put one new group of siblings on top, and answer it.

        Blank titles are dropped: a task with no name is a level of breakdown that says nothing, and what
        a tool refuses is a group where *every* title was blank — a call that
        wrote one real task and one accidental newline meant the one real task.
        """
        group = TodoGroup(items=[self._item(title, notes) for title in titles if title.strip()])
        self.groups.append(group)
        return group

    def pop(self) -> TodoGroup | None:
        """Take the TOP group off, and answer it — ``None`` when the stack was already empty."""
        return self.groups.pop() if self.groups else None

    def clear(self) -> int:
        """Empty the stack, and answer how many items went with it."""
        count = len(self.items())
        self.groups.clear()
        return count

    def update(
        self,
        group: int,
        number: int,
        *,
        title: str | None = None,
        status: TodoStatus | None = None,
        notes: str | None = None,
    ) -> TodoItem | None:
        """Change one item by its group's depth and its number inside that group.

        ``None`` means these two numbers name nothing — the caller says which of them was wrong,
        because only it knows what the listing looked like.

        A field left as ``None`` is a field the call did not mention: clearing a note is
        ``notes=""``, and nothing here can tell that apart from a call that said nothing about it.
        """
        found = self.group(group)
        if found is None or not 1 <= number <= len(found.items):
            return None
        item = found.items[number - 1]
        if title is not None:
            item.title = title.strip()
        if status is not None:
            item.status = status
        if notes is not None:
            item.notes = notes.strip()
        return item

    def pause(self, group: int, number: int) -> TodoItem | None:
        """Set one item aside, and answer it — ``None`` when it is not there or is set aside already.

        A method of its own because setting a task aside is not a field this call may or may not mention:
        it is the one transition the plan makes on the kernel's behalf, and a second write of it changes
        nothing (E188).
        """
        found = self.group(group)
        if found is None or not 1 <= number <= len(found.items):
            return None
        item = found.items[number - 1]
        if item.status is TodoStatus.PAUSED:
            return None
        item.status = TodoStatus.PAUSED
        return item

    def reminder(self) -> str:
        """What is left of the plan, as the one line a model is told between rounds.

        Empty when nothing is open **and nothing is set aside**, and that is a real answer: a reminder that
        has nothing to say is how a writer says the one it wrote before no longer holds (E142). A paused
        item is not work, so it is counted in a tail and never named among the open ones — a plan whose
        last open item was paused reads "nothing open (1 paused)", where silence would be half a lie.

        The titles are named up to :data:`NAMED` and counted past that, because this line is written once and
        then read in every request that follows it.
        """
        opened = self.open_items()
        paused = len(self.paused_items())
        tail = f" ({paused} paused)" if paused else ""
        if not opened:
            return f"nothing open{tail}" if paused else ""
        named = ", ".join(item.title for item in opened[:NAMED])
        rest = len(opened) - NAMED
        said = f"{named} and {rest} more" if rest > 0 else named
        return f"{counted(len(opened), 'item')} still open: {said}{tail}"

    def render(self, only: Collection[int] | None = None) -> str:
        """The listing: the whole stack, or the groups ``only`` names, each under its own number.

        ``only`` is a set of depths, which is how a tool that changed one layer answers with that
        layer and nothing else — and why "show the whole thing" is a choice only a listing makes: the
        other calls have no window to open in the first place.
        """
        if not self.groups:
            return EMPTY
        drawn = range(self.levels()) if only is None else sorted(d for d in only if 0 <= d < self.levels())
        return "\n".join(line for depth in drawn for line in self._layer(depth))

    def _layer(self, depth: int) -> list[str]:
        """One group's heading and its items, in the shape a model reads a checklist in."""
        heading = f"Group {depth} (TOP):" if depth == self.levels() - 1 else f"Group {depth}:"
        numbered = enumerate(self.groups[depth].items, start=1)
        lines = [f"  {MARKS[item.status]} {number}. {item.title}{_noted(item)}" for number, item in numbered]
        return [heading, *lines]

    def _item(self, title: str, notes: str) -> TodoItem:
        item = TodoItem(id=self.next_id, title=title.strip(), notes=notes.strip())
        self.next_id += 1
        return item

    def dump(self) -> str:
        """The whole stack as JSON text — one value, so reading it back is a fold and not a replay."""
        return msgspec.json.encode(self).decode()

    @classmethod
    def load(cls, text: str | None) -> Todos:
        """The stack a session wrote down, or an empty one when there is none this build can read.

        A value that does not decode is treated as nothing written: a conversation that cannot be
        opened at all is worse than one that forgot its plan.
        """
        if not text:
            return cls()
        try:
            return msgspec.json.decode(text, type=cls)
        except msgspec.DecodeError:
            return cls()


def _noted(item: TodoItem) -> str:
    """What was noted about a task, on the line it is already on."""
    return f" — {item.notes}" if item.notes else ""
