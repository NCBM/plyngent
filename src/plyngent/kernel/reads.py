"""Which lines of which files a turn has shown the model, and when it last looked.

``edit_lineno`` names lines by number, and a number is only worth acting on if the model has seen the
file as it is now. A range remembered from another file, from an earlier version of this one, or made
up from a summary would be an edit to lines nobody read — so a turn keeps what it has shown, per file,
with the file's modification time, and ``edit_lineno`` asks this record whether the range it was given
is one the model actually saw.

It belongs to the turn, because nothing else does: the
record has to end with the turn, and two turns of two conversations must not teach each other
anything. A turn's rounds share one, because a file read in round one is still read in round two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(slots=True, eq=False)
class Reads:
    """The lines this turn has shown, per file, and the file it showed them from."""

    _lines: dict[Path, set[int]] = field(default_factory=dict)
    _seen: dict[Path, int] = field(default_factory=dict)

    def mark(self, path: Path, lines: Iterable[int], *, seen: int) -> None:
        """Record that these 1-based lines of *path* were shown, as it was at ``seen`` (its mtime)."""
        self._lines.setdefault(path, set()).update(lines)
        self._seen[path] = seen

    def lines(self, path: Path) -> frozenset[int]:
        """The 1-based lines of *path* this turn has shown; empty when it has shown none."""
        return frozenset(self._lines.get(path, ()))

    def seen(self, path: Path) -> int | None:
        """The modification time the lines were read at, or ``None`` when nothing was read."""
        return self._seen.get(path)
