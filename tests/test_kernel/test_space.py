"""The pointer that says which space a member belongs to, read by a run that is standing in one.

No story can reach this alone: what a story sees is which tree a run opens, and that answer needs a recipe
line, a kernel to serve it and a member to stand in. What is left here is the file itself — where it lives,
what it says, and what happens when it says something no longer true (E193).
"""

from __future__ import annotations

from plyngent.kernel.journal import workspace_dir
from plyngent.kernel.space import SPACES, point, pointer_path, root_of, unpoint


def test_a_pointer_lives_beside_the_ledgers_and_is_named_the_same_way(tmp_path) -> None:
    member = tmp_path / "member"
    member.mkdir()

    pointer = pointer_path(member, data=tmp_path / "data")

    assert pointer.parent.name == SPACES
    assert pointer.parent.parent == tmp_path / "data", "under the data directory this program is told to use"
    assert pointer.name == workspace_dir(member, data=tmp_path / "data").name, "one naming rule for both"
    assert not pointer.exists(), "naming a pointer does not make one"


def test_a_member_says_which_tree_gathered_it(tmp_path) -> None:
    member = tmp_path / "member"
    root = tmp_path / "root"
    member.mkdir()
    root.mkdir()
    data = tmp_path / "data"

    assert root_of(member, data=data) is None, "nothing says anything about a tree nothing gathered"
    assert point(member, root, data=data) is True
    assert root_of(member, data=data) == root.resolve(), "the resolved root, which is what a kernel opens"

    assert unpoint(member, data=data) is True
    assert root_of(member, data=data) is None
    assert unpoint(member, data=data) is False, "forgetting what was never said says so"


def test_a_pointer_is_read_from_a_subdirectory_of_the_member_too(tmp_path) -> None:
    """A run starts wherever the shell left it, so the pointer is looked for up the tree (E193)."""
    member = tmp_path / "member"
    root = tmp_path / "root"
    deep = member / "src" / "inner"
    deep.mkdir(parents=True)
    root.mkdir()
    data = tmp_path / "data"

    assert point(member, root, data=data) is True
    assert root_of(deep, data=data) == root.resolve(), "the member's pointer answers for what is inside it"


def test_a_pointer_a_person_removed_is_the_ordinary_answer(tmp_path) -> None:
    """A tree with no pointer is one this run opens as itself, and that is not a failure to report."""
    member = tmp_path / "member"
    member.mkdir()
    pointer = pointer_path(member, data=tmp_path / "data")

    assert root_of(member, data=tmp_path / "data") is None, "an unreadable file and no file answer the same"

    pointer.parent.mkdir(parents=True)
    _ = pointer.write_text("\n", encoding="utf-8")
    assert root_of(member, data=tmp_path / "data") is None, "and so does an empty one"
