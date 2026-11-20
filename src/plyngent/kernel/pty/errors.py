"""What the PTY layer refuses, together with which kind of failure each refusal is.

The kind travels with the refusal so that no caller has to read a message and guess: whether a session
is gone, busy or past a limit changes both what the model is told and whether trying again could help,
and both of those come from one table (:mod:`plyngent.kernel.toolresult`). A raise site that does not
say defaults to ``BUG`` — an unclassified refusal is our own mistake, not the world's.

``PtyUnavailable`` is the same kind of thing one step out: no terminal *at all* on this machine, right
now — no broker, no platform support, no ``pywinpty``. It is the one refusal that is not about a session
and not about a call's arguments, which is why it is a class of its own; it is
still ``BLOCKED``, because a step is missing and nobody decided anything.
"""

from __future__ import annotations

from typing import override

from plyngent.kernel.failure import Failure


class PtyError(RuntimeError):
    """What the PTY layer refuses, and which kind of trouble it is."""

    failure: Failure

    def __init__(self, message: str, failure: Failure = Failure.BUG) -> None:
        super().__init__(message)
        self.failure = failure


class PtyUnavailable(PtyError):  # noqa: N818 — "unavailable" is the state, not a fault suffix
    """There is no terminal to be had here: no broker, no platform support, no ``pywinpty``.

    Raised, because there is nothing to return: the call did not start anything, and
    the reason is a sentence the caller shows. It is always ``BLOCKED``.
    """

    @override
    def __init__(self, message: str) -> None:
        super().__init__(message, Failure.BLOCKED)
