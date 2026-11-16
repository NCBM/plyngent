"""One host's numbers, and the window they can be asked about again.

None of this has a screen. A replayed event is rendered exactly like a live one, and what makes the two
different is which numbers arrive and in what order — which a story can only see by having two
connections and a client that remembers, so the arithmetic itself is pinned here.

What is asserted is the three rules a reconnect rests on: one count for the whole kernel, a window that
keeps the newest without inventing a number, and a mark this kernel did not make answering nothing.
"""

from __future__ import annotations

from plyngent.kernel.ledger import REPLAY_EVENTS, Ledger
from plyngent.wire.event import TurnStarted


def test_one_count_for_the_whole_kernel() -> None:
    """Every event gets the next number, whoever says it and however many connections are listening."""
    ledger = Ledger()

    said = [ledger.next(TurnStarted()) for _ in range(3)]

    assert [event.seq for event in said] == [1, 2, 3]
    assert ledger.seq == 3, "and the count is the last number it gave out"


def test_a_window_keeps_the_newest_and_answers_from_any_mark_inside_it() -> None:
    ledger = Ledger(kept=2)
    _ = [ledger.next(TurnStarted()) for _ in range(3)]

    assert [event.seq for event in ledger.since(0)] == [2, 3], "the newest, and never more than the window"
    assert [event.seq for event in ledger.since(2)] == [3], "a mark is read from, not including itself"
    assert ledger.since(3) == (), "nothing was said after the last thing said"


def test_a_mark_this_kernel_did_not_make_answers_nothing() -> None:
    """A number beyond the count is another kernel's, and replaying it would be replaying its words."""
    ledger = Ledger()
    _ = ledger.next(TurnStarted())

    assert ledger.since(REPLAY_EVENTS) == ()
    assert ledger.since(1) == ()
