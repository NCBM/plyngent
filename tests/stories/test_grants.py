"""What a conversation may reach outside its workspace: seen, and taken back.

The one question this program asks a person about a path hands out something that outlives the call that
asked it: for the rest of the conversation every tool resolves through it. A permission that cannot be
read is a fact somebody has to remember, and one that cannot be taken away is one that only ever grows —
so a read and a command, and the two of them are what these stories are about.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent_creative.model import calling, text

from plyngent.wire.command import Open
from plyngent.wire.query import Grants, GrantTable, Reached

from .driver import SESSION

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def _allowed_once(story_with: Scripted, tmp_path: Path) -> Story:
    """A conversation that has been allowed to read one directory outside its workspace."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _ = (outside / "notes.txt").write_text("kept outside", encoding="utf-8")
    story = await story_with(
        calling("listdir", {"path": "../elsewhere"}),
        calling(
            "request_directory_access",
            {"path": "../elsewhere", "mode": "read", "intention": "the notes are kept outside"},
        ),
        calling("listdir", {"path": "../elsewhere"}),
        text("done"),
    )
    await story.require("fs")
    await story.type("look outside")
    await story.answer("y")
    await story.finished()
    return story


async def test_what_a_conversation_may_reach_is_what_it_says_it_may_reach(story_with: Scripted, tmp_path: Path) -> None:
    """The read half: the table a person answered into is a fact a client can ask for."""
    story = await _allowed_once(story_with, tmp_path)
    other = await story.another_client()

    seen = await other.query(Grants(session=SESSION))

    assert isinstance(seen, GrantTable)
    assert seen.reached == (Reached(path=str(tmp_path / "elsewhere"), mode="read"),), (
        "the path the tool said it allowed, and the permission a person gave"
    )


async def test_a_permission_taken_back_is_one_the_next_call_does_not_have(story_with: Scripted, tmp_path: Path) -> None:
    """The other half: a "yes" that lasted the rest of the conversation can be ended before then."""
    story = await _allowed_once(story_with, tmp_path)
    other = await story.another_client()

    await story.type(f"/ungrant --path {tmp_path / 'elsewhere'}")
    await story.until("is no longer allowed")

    read_back = await other.query(Grants(session=SESSION))
    assert isinstance(read_back, GrantTable)
    assert read_back.reached == (), "the table is empty again, read over the connection that saw it full"


async def test_taking_back_what_was_never_allowed_says_so(story_with: Scripted, tmp_path: Path) -> None:
    """Two facts and not one: "gone now" and "was never there" are different things to be told."""
    story = await story_with(text("fine"))
    await story.type(f"/ungrant --path {tmp_path / 'nowhere'}")

    await story.until("was not allowed")


async def test_a_second_conversation_beside_it_is_allowed_nothing(story_with: Scripted, tmp_path: Path) -> None:
    """The unit of permission is a conversation: one "yes" answers for it and for no other."""
    story = await _allowed_once(story_with, tmp_path)
    other = await story.another_client()
    assert other.submit(Open(session="second")), "the connection was already gone, so the open went nowhere"
    await story.settle()

    theirs = await other.query(Grants(session="second"))
    mine = await other.query(Grants(session=SESSION))

    assert isinstance(theirs, GrantTable)
    assert theirs.reached == (), "a conversation opened beside it starts with nothing"
    assert isinstance(mine, GrantTable)
    assert mine.reached == (Reached(path=str(tmp_path / "elsewhere"), mode="read"),), (
        "and the conversation that asked keeps its row"
    )


async def test_a_person_can_ask_what_this_conversation_may_reach(story_with: Scripted, tmp_path: Path) -> None:
    """The word `/grants`, which is the whole reason the read exists: nobody has to remember a graph."""
    story = await _allowed_once(story_with, tmp_path)

    await story.type("/grants")
    await story.settle()

    assert f"{tmp_path / 'elsewhere'}: read" in story.notices(), "the table, said the way the tools say a path"
