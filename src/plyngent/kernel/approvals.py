"""What this host has already been told it may do, keyed by what a person means by "don't ask again".

A path is not a program and neither is a session: an approval to change one file says nothing about
another, and one to type into a live terminal cannot be scoped to a file. So every entry is a
:class:`~plyngent.kernel.toolspec.Remember` together with the subjects it covers — and nothing else: no
expiry, no inheritance from a directory to the files under it, and no widening from one call to the next.

**An entry also remembers how heavy the question was.** A yes is an answer to the question that was
put, so it covers that call and anything no heavier: agreeing to create a file in a directory does not
agree to replace one there, and the two arrive at the same path with the same posture. Without the
level in the table, "once per path" is one permission however it was asked for.

The table has one owner (the host) even though some of its contents are session-scoped. Two tables would
need a rule for *which one* asks, and a rule that answers "the session's" for one tool and "the host's"
for another is how a refusal becomes an approval nobody can find. A session-scoped approval is instead a
key that names its session: one table, no leak between sessions.

Nothing here asks anybody. This answers what is already agreed to; the executor is what puts a question
when the answer is no.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.toolspec import Posture
from plyngent.wire.ask import SEVERITIES, Severity

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import Remember

#: What an approval is scoped to: the kind of subject, and the subjects one call covered.
#:
#: The subjects are a tuple because one call may touch several — ``copy_path`` changes a source and a
#: destination, and an approval covering only one of them would let the other through unasked.
type Scoped = tuple[Remember, tuple[str, ...]]


class Approvals:
    """The approvals one host remembers, each with the level it was given at."""

    _once: dict[Scoped, Severity]

    def __init__(self) -> None:
        self._once = {}

    def needs(self, posture: Posture, *, remember: Remember | None, key: tuple[str, ...], severity: Severity) -> bool:
        """Whether this call has to be put to a person.

        ``AUTO`` never asks and ``ALWAYS`` always does — by definition, not by memory. ``ONCE`` asks
        exactly when this scope has not been agreed to *at least this heavily*: a yes covers the call
        that was put and anything lighter, and nothing heavier.
        """
        match posture:
            case Posture.AUTO:
                return False
            case Posture.ALWAYS:
                return True
            case _:
                if remember is None:  # `ToolSpec` refuses this combination; asking is the safe reading
                    return True
                granted = self._once.get((remember, key))
                return granted is None or _rank(granted) < _rank(severity)

    def agree(self, posture: Posture, *, remember: Remember | None, key: tuple[str, ...], severity: Severity) -> None:
        """Record that this call was allowed, under the scope it is remembered by and at its level.

        An ``ALWAYS`` approval is remembered by nothing on purpose: the word means the question is asked
        again next time, and recording it would quietly turn it into a ``ONCE``. A scope agreed to more
        heavily than this call keeps the heavier answer: what was allowed is not taken back by a
        lighter question being asked afterwards.
        """
        if posture is not Posture.ONCE or remember is None:
            return
        scoped = (remember, key)
        granted = self._once.get(scoped)
        if granted is None or _rank(severity) > _rank(granted):
            self._once[scoped] = severity


def _rank(severity: Severity) -> int:
    """Where a level sits in :data:`~plyngent.wire.ask.SEVERITIES`, so "at least this heavy" compares."""
    return SEVERITIES.index(severity)
