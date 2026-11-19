"""One kernel, several trees: which one a command means, and who hears what happens there.

A kernel serves a set of trees that grows while it runs, and one connection may work in any of them
(E154/E157). Two questions follow, and these stories are one each: a tree that has never been opened is
opened by naming it, and what happens in one tree is heard by the frontends watching *that* tree and by
nobody else.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from plyngent.wire.command import ListSessions, Open, Submit
from plyngent.wire.event import SessionsListed, TurnFinished
from plyngent.wire.query import Forest, Trajectory

from .driver import SESSION
from .model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from plyngent.wire.event import AnyEvent
    from plyngent.wire.link import KernelLink
    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


#: How long a lane this story holds directly is given to hand over what the kernel already said. Not a
#: story about a clock: a frontend with no screen has no ``until`` to wait on, and every one of these
#: reads a value the kernel has already emitted.
LANE_SECONDS = 5.0


async def _heard(link: KernelLink, want: type, *, wanted: int = 1) -> list[AnyEvent]:
    """Every event of one kind this connection has been told, waiting for at least ``wanted`` of them.

    The lane is drained and what is not wanted is dropped, because a frontend that reads its events is a
    frontend that has seen them: this is the only reader of ``link``'s lane in the story that calls it.
    """
    found: list[AnyEvent] = []
    while len(found) < wanted:
        event = await asyncio.wait_for(anext(link.events()), LANE_SECONDS)
        if isinstance(event, want):
            found.append(event)
    return found


async def test_a_tree_nobody_opened_is_opened_by_naming_it(story_with: Scripted, tmp_path: Path) -> None:
    """``Open(workspace=...)`` opens the tree, and the turn that names it runs *there*.

    The observation is the two journals: the conversation named ``s1`` in the second tree has rows, and the
    ``s1`` of the tree this program was started in has none. One id in two trees is two conversations, and
    which one a turn belongs to is what the workspace field is for.
    """
    second = tmp_path / "second"
    second.mkdir()
    _ = (second / "kept.txt").write_text("the other tree", encoding="utf-8")
    story = await story_with(calling("listdir", {"path": "."}), text("done"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(second)))
    other.submit(Submit(session="s1", workspace=str(second), text="look around"))
    told = await _heard(other, TurnFinished)

    assert told[0].workspace == str(second), "an event says which tree it happened in"
    assert told[0].session == "s1", "and which conversation in it"

    there = await other.query(Trajectory(session="s1", workspace=str(second)))
    here = await story.client.query(Trajectory(session=SESSION))

    assert isinstance(there, Forest)
    assert isinstance(here, Forest)
    assert there.entries, "the turn ran in the tree the command named"
    assert not here.entries, "and the conversation of that name in the other tree was not touched"


async def test_a_path_that_is_not_a_directory_is_refused_by_name(story: Story, tmp_path: Path) -> None:
    """A tree that is not a directory is answered, and the sentence says which path it was.

    Nothing is opened and nothing is guessed: a client that mistyped a path learns that, rather than
    getting a conversation in a tree it did not mean.
    """
    missing = tmp_path / "not-a-tree"

    story.client.submit(Open(session=SESSION, workspace=str(missing)))
    await story.until("workspace refused")

    assert any(str(missing) in line and "not a directory" in line for line in story.screen()), story.screen()


async def test_a_tree_inside_one_already_open_is_refused(story: Story) -> None:
    """Two trees over one set of files would change each other's work without knowing it (E154)."""
    inside = story.workspace / "inside"
    inside.mkdir()

    story.client.submit(Open(session=SESSION, workspace=str(inside)))
    await story.until("workspace refused")

    assert any("overlaps a tree that is already open" in line for line in story.screen()), story.screen()


async def test_a_command_naming_a_tree_nothing_opened_is_refused_rather_than_obeyed(
    story: Story, tmp_path: Path
) -> None:
    """Opening is ``Open``'s job: every other command naming a tree says so instead of opening one.

    A turn is the case that matters: a command that opened a tree behind the client's back would start
    writing a conversation somewhere nobody asked for.
    """
    second = tmp_path / "second"
    second.mkdir()

    story.client.submit(Submit(session=SESSION, workspace=str(second), text="hello"))
    await story.until("workspace refused")

    assert any("no tree is open here by that name" in line for line in story.screen()), story.screen()


async def test_a_listing_answers_for_the_tree_it_names(story_with: Scripted, tmp_path: Path) -> None:
    """``ListSessions`` is still per tree, so two trees with a conversation each are two answers and not one.

    The kernel grows no "all conversations" table: a client that wants to see every tree asks once per
    tree, which is what makes the answer a fact about a journal rather than about a process (E154).
    """
    second = tmp_path / "second"
    second.mkdir()
    story = await story_with(text("done"))
    other = await story.another_client()

    other.submit(Open(session="other", workspace=str(second)))
    other.submit(ListSessions(workspace=str(second)))
    other.submit(ListSessions(workspace=str(story.workspace)))
    listed = await _heard(other, SessionsListed, wanted=2)

    assert [row.session for row in listed[0].sessions] == ["other"], "the tree that was asked about"
    assert [row.session for row in listed[1].sessions] == [SESSION], "and the other one, asked about too"


async def test_two_frontends_on_one_tree_both_hear_it(story_with: Scripted) -> None:
    """A second frontend on one tree is a second view of one conversation, not a competitor for it.

    Both are told what happens: the screen this story types into draws the answer, and the connection the
    story holds directly is handed the same turn's events — which is what makes "our CLI draws one
    conversation" a choice of that CLI rather than a limit of the kernel (E154/E156).
    """
    story = await story_with(text("hello from the model"))
    other = await story.another_client()

    await story.type("say something")
    await story.finished()

    told = await _heard(other, TurnFinished)

    assert told[0].session == SESSION
    assert told[0].workspace == str(story.workspace), "and it says which tree it was in"
    assert any("hello from the model" in line for line in story.screen()), story.screen()


async def test_a_kernel_at_its_ceiling_refuses_one_more_tree(
    story_with: Scripted, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process holds a number of trees at once, and a client that asks for one more is told so.

    The ceiling is a constant — every tree carries a journal, a version store and its share of the
    terminals — and it is read where the opening happens, so a story can put it where it needs it (E157).
    """
    from plyngent import boot

    monkeypatch.setattr(boot, "MAX_TREES", 1)
    second = tmp_path / "second"
    second.mkdir()
    story = await story_with(text("done"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(second)))
    story.client.submit(Open(session=SESSION, workspace=str(second)))

    await story.until("workspace refused")

    assert any("as many trees open as it opens at once" in line for line in story.screen()), story.screen()
