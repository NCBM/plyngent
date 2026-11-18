"""What one conversation has already been told it may do, keyed by what a person means by "don't ask again".

A path is not a program and neither is a conversation: an approval to change one file says nothing about
another, and one to type into a live terminal cannot be scoped to a file. So every entry is a
:class:`~plyngent.kernel.toolspec.Remember` together with the subjects it covers — and nothing else: no
expiry, no inheritance from a directory to the files under it, and no widening from one call to the next.

**An entry also remembers how heavy the question was.** A yes is an answer to the question that was put,
so it covers that call and anything no heavier: agreeing to create a file in a directory does not agree to
replace one there, and the two arrive at the same path with the same posture. Without the level in the
table, "once per path" is one permission however it was asked for.

**The table belongs to the conversation that answered.** It used to be the host's, keyed by session id so
that one conversation's yes could not answer for another — which meant a second conversation in the same
tree was silently given the first one's answers (E156). Authorization is a conversation's (E153), so this
is a field of the session: a yes is a convenience for the conversation that gave it, held in memory for as
long as this process serves that conversation, and gone when the conversation is closed and opened again.
Asking again is the safe direction to be wrong in, and the one place where carrying answers over is worth
it — compaction, which folds a conversation into a new one — says so explicitly (:class:`Inherit`).

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
    """The approvals one conversation remembers, each with the level it was given at."""

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

    def inherit(self, earlier: Approvals) -> None:
        """Take every row *earlier* holds, keeping the heavier of two answers for one row.

        For the one caller that may ask: a new conversation seeded from an old one, with
        :class:`~plyngent.wire.command.Inherit.APPROVALS` on the command. A row that is already here is
        never made lighter — the merge is the same rule :meth:`agree` uses, applied twice over.
        """
        for scoped, granted in earlier._once.items():
            here = self._once.get(scoped)
            if here is None or _rank(granted) > _rank(here):
                self._once[scoped] = granted


def _rank(severity: Severity) -> int:
    """Where a level sits in :data:`~plyngent.wire.ask.SEVERITIES`, so "at least this heavy" compares."""
    return SEVERITIES.index(severity)
