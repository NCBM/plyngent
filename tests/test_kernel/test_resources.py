"""A discovery table is a list of roots and two callables: what a kind of resource costs to add.

Every test here invents its own kind. That is the point, not a convenience: the claim these
tests hold is that adding a kind of resource to a kernel is a ``scan`` and a list of roots, and a test
that used skills to say so would only ever prove something about skills.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from plyngent.kernel.resources import Clash, Conflict, Resources, Root

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Thing:
    """A kind of resource that is not in this kernel: what a discovered value looks like."""

    name: str
    label: str


def _scan(root: Root) -> Iterable[Thing]:
    """What one root holds: one file per value, so a table that caches would be caught out."""
    return (Thing(name=path.stem, label=root.label) for path in sorted(root.path.glob("*.md")))


def _table(*roots: Root, conflict: Conflict = Conflict.FIRST) -> Resources[Thing]:
    """The same kind, in a table with whatever conflict the test is about."""
    return Resources("thing", roots, conflict=conflict, scan=_scan, named=lambda thing: thing.name)


def _root(base: Path, origin: str, *names: str) -> Root:
    """A root holding these names, with the directory really made."""
    root = Root(path=base / origin, origin=origin, scope="user")
    root.path.mkdir(parents=True)
    for name in names:
        _ = (root.path / f"{name}.md").write_text(name, encoding="utf-8")
    return root


def test_the_first_root_to_hold_a_name_wins_it_and_the_copy_it_shadows_stays(tmp_path: Path) -> None:
    mine = _root(tmp_path, "mine", "x", "y")
    theirs = _root(tmp_path, "theirs", "x")
    table = _table(mine, theirs)

    assert [thing.name for thing in table.visible()] == ["x", "y"], "one per name, in name order"
    assert {thing.label for thing in table.visible()} == {"mine-user"}, "the earlier root wins both"
    assert len(table.all()) == 3, "the losing copy is still in the table — nothing is dropped silently"
    assert table.shadowed() == {"x": (Thing("x", "mine-user"), Thing("x", "theirs-user"))}, (
        "the winner first, then the copies it shadows"
    )


def test_priority_is_the_order_of_the_roots_and_nothing_else(tmp_path: Path) -> None:
    mine = _root(tmp_path, "mine", "x")
    theirs = _root(tmp_path, "theirs", "x")

    assert _table(theirs, mine).find("x") == Thing("x", "theirs-user"), "the list is the priority, read top to bottom"
    assert [thing.label for thing in _table(theirs, mine).all()] == ["theirs-user", "mine-user"]


def test_a_table_that_refuses_a_collision_raises_instead_of_picking(tmp_path: Path) -> None:
    table = _table(_root(tmp_path, "mine", "x"), _root(tmp_path, "theirs", "x"), conflict=Conflict.REFUSE)

    with pytest.raises(Clash) as raised:
        _ = table.visible()
    assert raised.value.names == ("x",), "the answer says which names are contested, not just that one is"
    assert table.clashes() == (("x", (Thing("x", "mine-user"), Thing("x", "theirs-user"))),), "and reports them"


def test_a_table_that_refuses_says_nothing_until_a_name_is_contested(tmp_path: Path) -> None:
    table = _table(_root(tmp_path, "mine", "x"), _root(tmp_path, "theirs", "y"), conflict=Conflict.REFUSE)

    assert [thing.name for thing in table.visible()] == ["x", "y"], "distinct names are not a configuration error"
    assert table.shadowed() == {}


def test_a_name_nothing_has_is_none_and_a_name_two_have_is_the_table_s_own_answer(tmp_path: Path) -> None:
    mine = _root(tmp_path, "mine", "x")
    theirs = _root(tmp_path, "theirs", "x")

    assert _table(mine, theirs).find("nope") is None, "a caller may name anything; finding nothing is an answer"
    assert _table(mine, theirs).find("x") == Thing("x", "mine-user"), "and a shadowed name resolves to the winner"
    with pytest.raises(Clash):
        _ = _table(mine, theirs, conflict=Conflict.REFUSE).find("x")


def test_a_root_with_no_directory_holds_nothing_and_is_reported_rather_than_failing(tmp_path: Path) -> None:
    missing = Root(path=tmp_path / "never-made", origin="mine", scope="user")
    table = _table(missing, _root(tmp_path, "theirs", "x"))

    assert table.absent() == (missing,), "a root that is not there is a fact a listing can say"
    assert [thing.name for thing in table.visible()] == ["x"]


def test_every_question_reads_the_roots_again(tmp_path: Path) -> None:
    mine = _root(tmp_path, "mine", "x")
    table = _table(mine)

    _ = (mine.path / "y.md").write_text("y", encoding="utf-8")

    assert [thing.name for thing in table.visible()] == ["x", "y"], "the roots are fixed; what is under them is not"
