"""The address a cancellation names, and the reasons it may carry.

Both are *values*: they are what a ``Cancel`` message spells, so they have to exist on both sides of
the boundary and neither side may own them. The mechanism that acts on a scope — a table of cancelled
prefixes, the work still running under them — stays in :mod:`plyngent.kernel.cancel`, because that is
the kernel's business and nothing outside it asks.

An address is a tuple of segments, for one reason: it can be written down,
parsed back, and created late. A scope that was born after its parent was cancelled still knows it is
cancelled, and a message carrying ``"3/r0/tool-1"`` reaches every scope below it without anything having
been registered in advance.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Self, final, override


class CancelReason(StrEnum):
    """Why a scope stopped.

    ``USER`` is a human, ``TIMEOUT`` an approval or a budget that ran out, ``POLICY`` a rule that
    refused, ``CLIENT_GONE`` a consumer that disconnected (which the boundary treats as a refusal), and
    ``LEAVING`` the session itself going down under a turn.
    """

    USER = "user"
    TIMEOUT = "timeout"
    POLICY = "policy"
    CLIENT_GONE = "client-gone"
    LEAVING = "leaving"


@final
@dataclass(frozen=True, slots=True)
class CancelScope:
    """One address under a turn: ``3/r0/tool-1/proc-0`` as ``("3", "r0", "tool-1", "proc-0")``."""

    path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.path:
            msg = "a cancel scope needs at least one segment"
            raise ValueError(msg)
        for segment in self.path:
            if not segment or "/" in segment:
                msg = f"a cancel scope segment must be non-empty and carry no '/': {segment!r}"
                raise ValueError(msg)

    @override
    def __str__(self) -> str:
        return "/".join(self.path)

    @classmethod
    def parse(cls, text: str) -> Self:
        """The scope ``text`` spells, in the form the boundary carries it."""
        return cls(tuple(text.split("/")))

    def child(self, *segments: str) -> Self:
        """This address extended by ``segments``."""
        return type(self)((*self.path, *segments))

    def under(self, other: Self) -> bool:
        """Whether this address is ``other`` or one of its descendants."""
        return self.path[: len(other.path)] == other.path
