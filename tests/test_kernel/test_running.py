"""How many turns one kernel runs at once, and what the one past the ceiling gets."""

from __future__ import annotations

import pytest

from plyngent.kernel.running import DEFAULT_RUNNING, Running, TooManyTurns


def test_four_turns_run_at_once_unless_the_file_says_otherwise() -> None:
    assert DEFAULT_RUNNING == 4
    assert Running().limit == 4


def test_the_one_past_the_ceiling_is_refused_and_not_queued() -> None:
    running = Running(2)

    running.start()
    running.start()

    assert running.now == 2
    with pytest.raises(TooManyTurns) as caught:
        running.start()
    assert caught.value.limit == 2, "the refusal carries the number, so a client can say both"
    assert running.now == 2, "and nothing was taken"


def test_a_turn_that_ends_gives_its_place_back() -> None:
    running = Running(1)

    running.start()
    running.stop()

    assert running.now == 0
    running.start()
    assert running.room is False


def test_a_turn_that_ends_twice_does_not_hand_back_two_places() -> None:
    running = Running(1)

    running.start()
    running.stop()
    running.stop()

    assert running.now == 0


def test_a_kernel_that_may_run_no_turns_is_not_a_kernel() -> None:
    with pytest.raises(ValueError, match="could never run one"):
        _ = Running(0)
