"""An OS error becomes a kind of failure in one place, so two tools cannot disagree about it."""

from __future__ import annotations

import errno

from plyngent.kernel.failure import _BY_ERRNO, Failure, Retry


def _raised(code: int) -> OSError:
    return OSError(code, errno.errorcode.get(code, "?"))


def test_one_errno_is_one_kind_wherever_it_is_raised() -> None:
    assert Failure.of(_raised(errno.ENOENT)) is Failure.MISSING
    assert Failure.of(_raised(errno.EACCES)) is Failure.PERMISSION
    assert Failure.of(_raised(errno.EBUSY)) is Failure.BUSY
    assert Failure.of(_raised(errno.ENOSPC)) is Failure.LIMIT
    assert Failure.of(_raised(errno.ETIMEDOUT)) is Failure.TIMEOUT


def test_each_number_belongs_to_exactly_one_kind() -> None:
    owners: dict[int, Failure] = {}
    for failure, codes in _BY_ERRNO.items():
        for code in codes:
            assert owners.setdefault(code, failure) is failure, f"{errno.errorcode[code]} is claimed twice"
            assert Failure.of(_raised(code)) is failure, "the lookup agrees with the table it was built from"


def test_an_errno_nobody_classified_is_reported_instead_of_guessed() -> None:
    assert Failure.of(_raised(errno.EILSEQ)) is Failure.BUG, "the table is the answer; a hole in it is ours"


def test_an_error_with_no_number_says_nothing_rather_than_guessing() -> None:
    assert Failure.of(OSError("something went wrong")) is Failure.BUG


def test_the_three_retries_are_the_three_next_moves() -> None:
    assert [retry.value for retry in Retry] == ["never", "same_call", "fix_args"]
