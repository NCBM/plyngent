"""One kind of discoverable thing, and the ordered plan of places to look for it.

Discovery is the same question for every kind of resource this kernel grows — a skill, and later a
plugin, a prompt template, an MCP server: *which of these are there, and where did each of them come
from*. So it is one value, and what differs between kinds is two callables and a list of roots: this
module knows about roots and priority, and nothing at all about what is being found.

**Priority is the order of the roots, and there is no number for it.** A ``priority: int`` field is a
list that admits two entries with one rank, and the answer to "what happens then" is whatever the sort
did last — a fact about the sorting, not about the design. A sequence cannot say it. So the first root
to hold a name wins it, the copies it shadows stay in the table, and nothing is silently dropped: a
skill that lost a collision is exactly what somebody needs to see before deleting it.

**Two ways to answer a collision are two kinds of table.** ``FIRST`` is the ordinary one — the more
specific root wins, as a workspace's conventions win over a user's. ``REFUSE`` is for a kind where two
roots holding one name is a configuration nobody could have meant, and where picking one would hide a
mistake; it answers by raising :class:`Clash`.

Nothing here caches: every question re-reads the roots, and the roots are the only thing that is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path


class Conflict(StrEnum):
    """What a table does when two roots hold one name.

    ``FIRST`` keeps both copies and lets the earlier root's win. ``REFUSE`` says the table cannot be
    read at all until somebody fixes it — for a kind where two copies are never what a person meant.
    """

    FIRST = "first"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class Root:
    """One directory to look in, and where it came from.

    ``origin`` is who put the directory there (``plyngent`` for our own, ``claude`` for another
    harness's) and ``scope`` is how far it applies (``user``, ``project``). Together they are the label
    a listing shows, so a model reading two copies of a name can tell which one it is reading.
    """

    path: Path
    origin: str
    scope: str

    @property
    def label(self) -> str:
        """``<origin>-<scope>``, e.g. ``claude-project``."""
        return f"{self.origin}-{self.scope}"


class Clash(ValueError):  # noqa: N818 — a table refusing to pick is its answer, not a fault
    """Two roots hold one name, in a table whose :class:`Conflict` says it may not pick one."""

    kind: str
    names: tuple[str, ...]

    def __init__(self, kind: str, names: tuple[str, ...]) -> None:
        super().__init__(f"{kind}: more than one root holds {', '.join(names)}, and this table refuses to pick")
        self.kind = kind
        self.names = names


class Resources[T]:
    """Everything of one kind under an ordered list of roots.

    ``scan`` reads one root and yields what is under it, ``named`` says what a scanned value is called.
    Both belong to the kind: a new kind of resource is a ``scan`` and a list
    of roots, and nothing in the discovery below has to be told that it exists.
    """

    kind: str
    roots: tuple[Root, ...]
    _conflict: Conflict
    _scan: Callable[[Root], Iterable[T]]
    _named: Callable[[T], str]

    def __init__(
        self,
        kind: str,
        roots: Sequence[Root],
        *,
        conflict: Conflict,
        scan: Callable[[Root], Iterable[T]],
        named: Callable[[T], str],
    ) -> None:
        self.kind = kind
        self.roots = tuple(roots)
        self._conflict = conflict
        self._scan = scan
        self._named = named

    def all(self) -> tuple[T, ...]:
        """Every value under every root, in priority order, shadows included."""
        return tuple(value for root in self.roots for value in self._scan(root))

    def visible(self) -> tuple[T, ...]:
        """One value per name, in name order: the first root to claim a name wins it."""
        by_name = self._by_name()
        self._refuse(by_name)
        return tuple(by_name[name][0] for name in sorted(by_name))

    def shadowed(self) -> dict[str, tuple[T, ...]]:
        """The names more than one root holds, with every copy in priority order — the winner first."""
        return {name: copies for name, copies in sorted(self._by_name().items()) if len(copies) > 1}

    def clashes(self) -> tuple[tuple[str, tuple[T, ...]], ...]:
        """What a refusing table cannot resolve, as rows: the name, and every copy of it."""
        return tuple(self.shadowed().items())

    def find(self, name: str) -> T | None:
        """The visible value called *name*, or ``None`` — a caller may name anything."""
        by_name = self._by_name()
        copies = by_name.get(name)
        if copies is None:
            return None
        self._refuse(by_name)
        return copies[0]

    def absent(self) -> tuple[Root, ...]:
        """The roots with no directory under them.

        A root that is not there holds nothing, which is not an error: the default write root does not
        exist until the first thing is written into it, and a listing says so.
        """
        return tuple(root for root in self.roots if not root.path.is_dir())

    def _by_name(self) -> dict[str, tuple[T, ...]]:
        """Every value, grouped by name, each group in priority order."""
        found: dict[str, list[T]] = {}
        for value in self.all():
            found.setdefault(self._named(value), []).append(value)
        return {name: tuple(copies) for name, copies in found.items()}

    def _refuse(self, by_name: Mapping[str, tuple[T, ...]]) -> None:
        """Raise when this table refuses to pick, and there is something to pick between."""
        if self._conflict is Conflict.REFUSE:
            contested = tuple(sorted(name for name, copies in by_name.items() if len(copies) > 1))
            if contested:
                raise Clash(self.kind, contested)
