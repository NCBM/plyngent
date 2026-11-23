"""One kernel, several trees: which one a command means, who hears what happens there, and what closing one does.

A kernel serves a set of trees that grows while it runs, and one connection may work in any of them
(E154/E157). Three questions follow, and these stories are one each: a tree that has never been opened is
opened by naming it, what happens in one tree is heard by the frontends watching *that* tree and by nobody
else, and a tree a command closes goes away with the conversations in it while its history stays (E181).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from plyngent.wire.command import CloseWorkspace, ListSessions, Open, Submit
from plyngent.wire.event import (
    Refusal,
    SessionsListed,
    TurnFinished,
    TurnStarted,
    WorkspaceClosed,
    WorkspaceRefused,
)
from plyngent.wire.query import Forest, Trajectory

from .driver import SESSION
from .model import Held, calling, text

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

    Nothing is opened and nothing is guessed: a client that mistyped a path learns that, and gets no
    conversation in a tree it did not mean.
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
    """Opening is ``Open``'s job: every other command naming a tree says so, and opens nothing.

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
    tree, which is what makes the answer a fact about a journal, not about a process (E154).
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
    conversation" a choice of that CLI, not a limit of the kernel (E154/E156).
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


async def test_a_closed_tree_takes_its_conversations_and_every_connection_hears_it(
    story_with: Scripted, tmp_path: Path
) -> None:
    """``CloseWorkspace`` ends what is open in a tree, and the snapshot others were greeted with is corrected.

    Told to every connection and not only to the one that asked: a tree a client never touched is the one
    whose next command would otherwise name something a greeting still listed. What the event carries is the
    tree, and the conversations that went with it (E181).
    """
    second = tmp_path / "second"
    second.mkdir()
    story = await story_with(text("done"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(second)))
    other.submit(CloseWorkspace(workspace=str(second)))
    told = await _heard(other, WorkspaceClosed)

    assert told[0].workspace == str(second), "the tree that is gone"
    assert told[0].sessions == ("s1",), "and the conversations that went with it"
    await story.until(f"closed {second}")

    assert any("it held s1" in line for line in story.screen()), story.screen()


async def test_a_tree_nothing_opened_is_not_closed(story: Story, tmp_path: Path) -> None:
    """Closing names a tree somebody opened, and a path nothing has open is refused for the reason opening is."""
    never = tmp_path / "never"
    never.mkdir()

    story.client.submit(CloseWorkspace(workspace=str(never)))
    await story.until("workspace refused")

    assert any("no tree is open here by that name" in line for line in story.screen()), story.screen()


async def test_a_tree_with_a_turn_in_flight_is_not_closed_until_it_ends(story_with: Scripted, tmp_path: Path) -> None:
    """A refusal that clears itself: a tree whose own work has not finished is closed later, and nothing else.

    The count it reads belongs to the tree's host, so a turn running in another tree is not this tree's
    business — the second close is answered with the tree closing (E181).
    """
    second = tmp_path / "second"
    second.mkdir()
    held = Held(calling("listdir", {"path": "."}))
    story = await story_with(held, text("done"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(second)))
    other.submit(Submit(session="s1", workspace=str(second), text="look around"))
    _ = await _heard(other, TurnStarted)
    other.submit(CloseWorkspace(workspace=str(second)))
    refused = await _heard(other, WorkspaceRefused)
    assert refused[0].reason is Refusal.TURN_RUNNING, "a tree that is still being written is not closed"

    held.ready.set()
    _ = await _heard(other, TurnFinished)
    other.submit(CloseWorkspace(workspace=str(second)))
    told = await _heard(other, WorkspaceClosed)

    assert told[0].workspace == str(second), "and the same command made later is obeyed"


async def test_a_closed_tree_makes_room_for_the_one_it_held(story_with: Scripted, tmp_path: Path) -> None:
    """The nesting refusal reads the table that is open, so a tree closed is a tree that no longer blocks one.

    ``inner`` is inside ``outer``, which is why it is refused while ``outer`` is up: two trees over one set of
    files would change each other's work without knowing it (E154/E157). Closing the one on the outside is
    what makes the one inside a tree this kernel can open.
    """
    outer = tmp_path / "outer"
    inner = outer / "inner"
    inner.mkdir(parents=True)
    story = await story_with(text("done"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(outer)))
    other.submit(Open(session="s1", workspace=str(inner)))
    nested = await _heard(other, WorkspaceRefused)
    assert nested[0].reason is Refusal.NESTED_WORKSPACE

    other.submit(CloseWorkspace(workspace=str(outer)))
    _ = await _heard(other, WorkspaceClosed)
    other.submit(Open(session="s1", workspace=str(inner)))
    other.submit(ListSessions(workspace=str(inner)))
    listed = await _heard(other, SessionsListed)

    assert [row.session for row in listed[0].sessions] == ["s1"], "the tree inside the closed one is open"


async def test_a_tree_that_was_closed_opens_again_as_the_same_conversation(
    story_with: Scripted, tmp_path: Path
) -> None:
    """A closed tree keeps its journal, so opening it again is that conversation and not a second one.

    Closing is about what this process is holding, never about what is on disk: the entries are there, and
    the same id in the same tree is the same conversation (E181).
    """
    second = tmp_path / "second"
    second.mkdir()
    story = await story_with(calling("listdir", {"path": "."}), text("the first answer"))
    other = await story.another_client()

    other.submit(Open(session="s1", workspace=str(second)))
    other.submit(Submit(session="s1", workspace=str(second), text="look around"))
    _ = await _heard(other, TurnFinished)
    before = await other.query(Trajectory(session="s1", workspace=str(second)))

    other.submit(CloseWorkspace(workspace=str(second)))
    _ = await _heard(other, WorkspaceClosed)
    other.submit(Open(session="s1", workspace=str(second)))
    other.submit(ListSessions(workspace=str(second)))
    _ = await _heard(other, SessionsListed)
    after = await other.query(Trajectory(session="s1", workspace=str(second)))

    assert isinstance(before, Forest) and isinstance(after, Forest)
    assert before.entries, "the turn left rows the first time"
    assert [entry.id for entry in after.entries] == [entry.id for entry in before.entries], (
        "the same conversation, read back from the same journal"
    )


async def test_a_tree_opened_after_a_close_is_addressed_as_a_new_one(story_with: Scripted, tmp_path: Path) -> None:
    """A qualifier names sessions for as long as this process runs, so closing a tree may not free its name.
    Three trees, the middle one closed, a fourth opened: if the fourth took the closed one's number it would
    be the *same* address as the tree that still holds it, and one conversation would be read as another's.
    The third tree's entries are what proves it — its conversation is still its own (E181).
    """
    story = await story_with(text("the third tree's answer"))
    other = await story.another_client()
    middle = tmp_path / "middle"
    third = tmp_path / "third"
    fourth = tmp_path / "fourth"
    for one in (middle, third, fourth):
        one.mkdir()

    other.submit(Open(session="s1", workspace=str(middle)))
    other.submit(Open(session="s1", workspace=str(third)))
    other.submit(Submit(session="s1", workspace=str(third), text="look around"))
    _ = await _heard(other, TurnFinished)

    other.submit(CloseWorkspace(workspace=str(middle)))
    _ = await _heard(other, WorkspaceClosed)
    other.submit(Open(session="s1", workspace=str(fourth)))
    other.submit(ListSessions(workspace=str(fourth)))
    _ = await _heard(other, SessionsListed)

    there = await other.query(Trajectory(session="s1", workspace=str(third)))
    here = await other.query(Trajectory(session="s1", workspace=str(fourth)))

    assert isinstance(there, Forest) and isinstance(here, Forest)
    assert there.entries, "the third tree's conversation is still its own"
    assert not here.entries, "and the tree opened after the close started with nothing"


async def test_a_client_does_not_close_the_tree_it_is_standing_in(story_with: Scripted) -> None:
    """The kernel would obey, and this frontend is the one that says no: it is drawing that tree (E181).

    The word is the kernel's, so it is listed and it describes itself. What a person typing it here gets is a
    sentence from this program, and the conversation they are in goes on as though nothing had been asked: a
    frontend that closed its own tree would have nothing left to draw.
    """
    story = await story_with(text("still here"))

    await story.type("/close-workspace")

    (said,) = story.notices()
    assert said.startswith("this client is standing in "), said
    assert story.workspace.name in said, "and it says which tree it means"
    assert said.endswith(", so it is not the one to close it"), said
    await story.type("are you still there")
    await story.finished()

    assert any("still here" in line for line in story.screen()), "and the tree it stands in is still serving"
