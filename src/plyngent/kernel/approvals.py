"""What this host has already been told it may do, keyed by what a person means by "don't ask again".

A path is not a program and neither is a session: an approval to change one file says nothing about
another, and one to type into a live terminal cannot be scoped to a file. So every entry is a
:class:`~plyngent.kernel.toolspec.Remember` together with the subjects it covers — and nothing else: no
expiry, no inheritance from a directory to the files under it, and no widening from one call to the next.

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

if TYPE_CHECKING:
    from plyngent.kernel.toolspec import Remember

#: One remembered approval: the kind of subject it is scoped to, and the subjects it covers.
#:
#: The subjects are a tuple because one call may touch several — ``copy_path`` changes a source and a
#: destination, and an approval covering only one of them would let the other through unasked.
type Agreed = tuple[Remember, tuple[str, ...]]


class Approvals:
    """The approvals one host remembers, and the one question asked about them."""

    _once: set[Agreed]

    def __init__(self) -> None:
        self._once = set()

    def needs(self, posture: Posture, *, remember: Remember | None, key: tuple[str, ...]) -> bool:
        """Whether this call has to be put to a person.

        ``AUTO`` never asks and ``ALWAYS`` always does — by definition, not by memory. ``ONCE`` asks
        exactly when this key is not in the table, which is the whole content of the word "once".
        """
        match posture:
            case Posture.AUTO:
                return False
            case Posture.ALWAYS:
                return True
            case _:
                if remember is None:  # `ToolSpec` refuses this combination; asking is the safe reading
                    return True
                return (remember, key) not in self._once

    def agree(self, posture: Posture, *, remember: Remember | None, key: tuple[str, ...]) -> None:
        """Record that this call was allowed, under the scope it is remembered by.

        An ``ALWAYS`` approval is remembered by nothing on purpose: the word means the question is asked
        again next time, and recording it would quietly turn it into a ``ONCE``.
        """
        if posture is Posture.ONCE and remember is not None:
            self._once.add((remember, key))
