"""The one thing the kernel may ask a program's configuration to do: write a route's models back.

The kernel is not allowed to know how it was configured — a file is read by :func:`plyngent.config.load`,
and what reaches the kernel is values — so a write-back is a *narrow door*: one method, handed in where
the host is assembled, implemented where files are read (E173). A frontend that edited the file itself
would be a second place deciding what this program is, and a frontend that was told to write one on
another machine could not do it at all.

**What is not here is the point.** There is no setter for a key, no path a caller may name and no
document a caller may hold: "what can be written" is the same list as "which command writes it", and that
list has one entry. A general ``set(key, value)`` would make ``[sandbox]`` writable — the line drawn
before anybody was asked — along with the ``[plugins]`` list, which only a person adds to, and
``[journal].path``, which this process has already read and could not start honouring.

**A failed write leaves the file exactly as it was.** The four gates and what each of them says about the
file live where the file is: this module only names the door and the ways a closing door is described.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from plyngent.kernel.lm.route import ModelEntry
    from plyngent.wire.event import PersistProblem


@dataclass(frozen=True, slots=True)
class Written:
    """What one write-back did: the file it left whole, or the reason it left the file alone."""

    path: str = ""
    problem: PersistProblem | None = None


class Persisting(Protocol):
    """A configuration that can be written back, as the kernel is allowed to see it."""

    def write_models(self, route: str, models: tuple[ModelEntry, ...]) -> Written:
        """Write *models* as *route*'s table, and answer what became of the file.

        ``problem`` is ``None`` when the file now says what the kernel holds; anything else is the reason
        it does not, and the file is untouched.
        """
        ...
