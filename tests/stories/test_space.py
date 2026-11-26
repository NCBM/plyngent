"""One tree that reaches another: a person gathers a member, a turn reads in it, and letting it go closes it.

A space is a root tree plus the trees it gathered (E193). What a story can see, and what no unit can, is
the whole shape from outside: the line a person types, the tools of a turn reaching a member's directory
without a question being asked about it, and the member going away again as one more row in the root's own
file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.wire.command import Open, Space
from plyngent.wire.event import SpaceVerb
from plyngent_creative.model import calling, text

from .driver import SESSION

if TYPE_CHECKING:
    from pathlib import Path


async def test_a_member_is_as_reachable_as_the_root_and_no_question_is_asked(story_with, tmp_path: Path) -> None:
    """Gathering a tree makes its directory reachable, and the turn that reads there is not stopped.

    Three observations: the table the command answers with, the text the tool read from a path outside the
    root, and the questions a person was asked — none, because a tree somebody put in the space by name is
    a tree they have already allowed (E193).
    """
    member = tmp_path / "member"
    member.mkdir()
    _ = (member / "kept.txt").write_text("said in the member", encoding="utf-8")
    story = await story_with(calling("read", {"path": str(member / "kept.txt")}), text("done"))

    await story.type(f"/space add --path {member}")
    await story.until("this space reaches:")

    assert any("gathered" in line and str(member) in line for line in story.screen()), story.screen()

    await story.type("what is in the member")
    await story.finished()

    assert any("kept.txt (lines 1-1 of 1)" in line for line in story.screen()), "the read reached the member"
    assert not any("escapes workspace root" in line for line in story.screen()), "and nothing refused it"
    assert story.questions() == (), "a member is not a path a person has to allow"


async def test_a_member_is_not_a_tree_a_client_may_address(story_with, tmp_path: Path) -> None:
    """A member is reached through its root, so opening it as a tree of its own is the second address refused."""
    member = tmp_path / "member"
    member.mkdir()
    story = await story_with(text("done"))

    await story.type(f"/space add --path {member}")
    await story.until("this space reaches:")

    story.client.submit(Open(session=SESSION, workspace=str(member)))
    await story.until("workspace refused")

    assert any(str(member) in line for line in story.screen() if "workspace refused" in line), story.screen()


async def test_letting_a_tree_go_puts_the_path_back_outside(story_with, tmp_path: Path) -> None:
    """Withdrawing a member is one more row, and the next call about that path is refused again.

    Reversible in both directions, which is the half a one-way grant would have got wrong: what a person
    gathered by name is what a person can take back (E193).
    """
    member = tmp_path / "member"
    member.mkdir()
    _ = (member / "kept.txt").write_text("said in the member", encoding="utf-8")
    story = await story_with(calling("read", {"path": str(member / "kept.txt")}))

    await story.type(f"/space add --path {member}")
    await story.until("this space reaches:")
    await story.type(f"/space drop --path {member}")
    await story.until("let go")

    assert any("let go" in line and str(member) in line for line in story.screen()), story.screen()
    assert any("no other tree" in line for line in story.screen()), "the table says nothing is reached any more"


async def test_a_tree_inside_one_the_space_reaches_is_refused_by_name(story_with) -> None:
    """Two trees over one set of files would change each other's work, so overlap is answered, not allowed."""
    story = await story_with(text("done"))
    inside = story.workspace / "inside"
    inside.mkdir()

    story.client.submit(Space(verb=SpaceVerb.ADD, path=str(inside)))
    await story.until("workspace refused")

    assert any("overlaps a tree this space already reaches" in line for line in story.screen()), story.screen()


async def test_a_space_answers_with_the_table_and_not_with_a_receipt(story_with) -> None:
    """``list`` and ``add`` are one answer, because a space is one table and nobody keeps a second copy."""
    story = await story_with(text("done"))

    await story.type("/space")
    await story.until("this space reaches: no other tree")


async def test_a_tree_nothing_gathered_is_nothing_to_let_go_of(story_with, tmp_path: Path) -> None:
    """A path this space never reached is answered as the fact it is, and the table does not move."""
    loose = tmp_path / "loose"
    loose.mkdir()
    story = await story_with(text("done"))

    story.client.submit(Space(verb=SpaceVerb.DROP, path=str(loose)))
    await story.until("workspace refused")

    assert any("nothing to let go of" in line for line in story.screen()), story.screen()
