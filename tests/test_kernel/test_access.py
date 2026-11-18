"""A conversation's table is the only grant table, and one row answers for a whole subtree."""

from __future__ import annotations

from plyngent.kernel.access import MAX_GRANTS, AccessMode, covers


def test_a_row_answers_for_what_is_under_it_and_nothing_else(tmp_path) -> None:
    shared = tmp_path / "shared"

    assert covers({shared: AccessMode.READ}, shared, AccessMode.READ)
    assert covers({shared: AccessMode.READ}, shared / "a.txt", AccessMode.READ)
    assert not covers({shared: AccessMode.READ}, shared / "a.txt", AccessMode.WRITE)
    assert not covers({shared: AccessMode.READ}, tmp_path / "elsewhere", AccessMode.READ)


def test_a_table_with_no_rows_answers_nothing(tmp_path) -> None:
    assert not covers({}, tmp_path, AccessMode.READ)
    assert not covers(None, tmp_path, AccessMode.READ), "a caller before its conversation exists has no table"


def test_the_widest_of_the_matching_rows_is_the_answer(tmp_path) -> None:
    shared = tmp_path / "shared"
    nested = shared / "nested"

    assert covers({shared: AccessMode.READ, nested: AccessMode.EXEC}, nested / "run.sh", AccessMode.EXEC)
    assert covers({shared: AccessMode.EXEC, nested: AccessMode.READ}, nested / "run.sh", AccessMode.WRITE), (
        "a narrower row does not take away a wider one above it"
    )


def test_one_table_holds_a_bounded_number_of_rows() -> None:
    assert MAX_GRANTS == 32
