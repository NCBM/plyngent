"""How many turns this host is running, and the ceiling the next one has to fit under.

Turns run beside each other because a conversation is the unit of work, not the process: four frontends on
one kernel, or one frontend with four conversations, each waiting on its own model. What a machine cannot
do is run an unbounded number of them, so there is a number — `[turn] running`, four by default — and a
fifth turn is refused rather than queued.

**Refused, not queued.** A queue would make "how many turns are running" a number that is not the work in
flight: the frontend that asked would be told its turn started and would then wait behind turns it cannot
see, and the read that answers "what is running, right now" would be a lie about what is being done. A
refusal is a value the client can render and retry from, and retrying is its business — the same rule
every other refusal here follows (E156).
"""

from __future__ import annotations

#: How many turns a kernel runs at once unless `[turn] running` says otherwise. Four is one conversation
#: per frontend for the ordinary case, and few enough that a machine is not asked for a dozen model
#: requests at the same instant.
DEFAULT_RUNNING = 4


class TooManyTurns(Exception):  # noqa: N818 — a refusal, not a fault in the caller
    """A turn was not started because this host is already running as many as it may.

    ``limit`` travels on the exception because the refusal says both numbers: the one that was asked for,
    and the one this kernel will run.
    """

    limit: int

    def __init__(self, limit: int) -> None:
        super().__init__(f"this kernel is already running {limit} turns, which is the limit")
        self.limit = limit


class Running:
    """The turns one host is running, and how many it may run at once."""

    limit: int
    _now: int

    def __init__(self, limit: int = DEFAULT_RUNNING) -> None:
        if limit < 1:
            msg = f"a kernel that may run {limit} turns at once could never run one"
            raise ValueError(msg)
        self.limit = limit
        self._now = 0

    @property
    def now(self) -> int:
        """How many turns are running."""
        return self._now

    @property
    def room(self) -> bool:
        """Whether one more turn fits."""
        return self._now < self.limit

    def start(self) -> None:
        """Take one of the places, or refuse because there are none."""
        if self._now >= self.limit:
            raise TooManyTurns(self.limit)
        self._now += 1

    def stop(self) -> None:
        """Give one back; a turn that ends twice must not hand back a place it never took."""
        self._now = max(0, self._now - 1)
