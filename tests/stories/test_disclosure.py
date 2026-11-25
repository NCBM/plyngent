"""What a session is offered, and the one call that adds to it.

A conversation starts with the tools it cannot work without and gains the rest by asking, one group at
a time. These stories are about the two ends of that: the block and the line a model is given on the
first turn, and what happens when it asks for a group — or calls something it was never shown.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from plyngent.cli.offers import ClientTools
from plyngent.kernel.model import Call, Reply, Result
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text

#: A client that can run nothing. An offer moves every conversation's tool set when it arrives, and a
#: story about the number a journal left behind wants it to be nobody else's.
SILENT = ClientTools(offer=(), handlers={})

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request

    type Scripted = Callable[..., Awaitable[Story]]


#: The names `tool_require(group="fs")` shows, in the order the table folds them: a story that says what
#: arrived says all of it, because a listing that lost one would still pass a check for a prefix.
FS_TOOLS = "copy_path, delete_path, listdir, move_path, new_temporary_workspace, request_directory_access, tree"


def _asked(tool: str, *, call: str = "", **arguments: object) -> Call:
    """One call of a batch: the model's own id for it, and its arguments as the model would spell them."""
    return Call(id=call or tool, tool=tool, arguments=json.dumps(arguments))


def _batch(*calls: Call) -> Reply:
    """One answer that asks for several calls, in the order the model wrote them."""
    return Reply(calls=calls)


def _names(request: Request) -> set[str]:
    """The tools one request offered the model."""
    return {tool.name for tool in request.tools}


def _told_of(request: Request) -> str:
    """What the model was told at the end of one round: the result of the call that round answered."""
    said = request.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said.text


def _index(request: Request) -> str:
    """The last section of the prompt: what this session has not been shown, and whether anyone can be asked.

    The prompt is five sections and this is the one the tool set writes (§0.10, E63), so a story about
    disclosure reads it, and not the playbook that precedes it.
    """
    return request.system.rsplit("\n\n", 1)[-1]


async def test_asking_for_a_group_brings_its_tools_in_the_next_round(story_with: Scripted) -> None:
    story = await story_with(
        calling("tool_require", {"group": "fs"}),
        calling("listdir", {"path": "."}),
        text("done"),
    )
    _ = (story.workspace / "notes.txt").write_text("hi", encoding="utf-8")

    await story.type("what can you do with files?")
    await story.finished()

    first, second, last = story.requests()
    assert "listdir" not in _names(first), "a group nobody asked for is not offered"
    assert _index(first) == (
        "7 more groups (22 tools): code(1), document(1), fs(7), net(1), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one"
    ), "and the line says what is missing, because a group nobody mentions cannot be asked for"
    assert "listdir" in _names(second), "what one round asks for arrives in the next"
    assert _index(second) == (
        "6 more groups (15 tools): code(1), document(1), net(1), process(6), session(1), vcs(5) — "
        "use tool_require(group=…) to add one"
    ), "a group that has arrived is not still described as missing"
    assert f"tool_require ok: showing 7 tools from fs: {FS_TOOLS}" in (story.screen()), (
        "the person and the model are told what arrived, by name"
    )
    assert "file\tnotes.txt" in _told_of(last), "and the tool it bought works in that same round"


async def test_a_tool_in_a_group_nobody_was_shown_is_answered_with_its_group(story_with: Scripted) -> None:
    story = await story_with(calling("listdir", {"path": "."}), text("fair"))
    _ = (story.workspace / "notes.txt").write_text("hi", encoding="utf-8")

    await story.type("what is here?")

    assert _told_of(story.requests()[-1]) == (
        "'listdir' is not available yet: it is in the 'fs' group\nhint: tool_require(group=fs) — show that group first"
    ), "calling an unshown tool is answered with the one call that shows it, and nothing ran"
    assert not any(line.startswith("file\t") for line in story.screen())


async def test_two_asks_for_one_group_cost_one_change_of_the_tool_set(story_with: Scripted) -> None:
    story = await story_with(
        _batch(_asked("tool_require", call="first", group="fs"), _asked("tool_require", call="second", group="fs")),
        text("good"),
    )

    await story.type("I will need the file tools")
    await story.finished()

    assert story.toolset_epoch == 2, "the client's offer was the first change; asking twice is the second"
    (_, second) = story.requests()
    answers = [message.text.splitlines()[0] for message in second.messages if isinstance(message, Result)]
    assert answers == [
        f"showing 7 tools from fs: {FS_TOOLS}",
        "fs was already shown",
    ]


async def test_a_group_that_does_not_exist_lists_the_ones_that_do(story_with: Scripted) -> None:
    story = await story_with(calling("tool_require", {"group": "nope"}), text("right"))

    await story.type("show me everything")

    assert _told_of(story.requests()[-1]) == (
        "no group named 'nope'; there are: code, document, fs, net, process, session, vcs"
    ), "a wrong name is an argument error that can be fixed, so the answer says what the names are"


async def test_a_hint_names_the_tool_once_its_group_is_shown(story_with: Scripted) -> None:
    """The other half of the fallback: with the group shown, the same answer gives the call that works."""
    story = await story_with(calling("read", {"path": "sub"}), text("right"))
    _ = (story.workspace / "sub").mkdir()
    await story.require("fs")

    await story.type("read the directory")
    await story.finished()

    assert _told_of(story.requests()[-1]).splitlines() == [
        "not a file: sub",
        "hint: listdir(path=sub) — see what is in it",
    ], "a hint may name a tool only when the tool is one the model was offered"


async def test_a_group_a_conversation_asked_for_is_still_shown_after_a_restart(tmp_path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("tool_require", {"group": "fs"}), text("noted")]),
        journal=journal,
    )
    try:
        await first.type("I will need the file tools")
        assert any(line.startswith("tool_require ok: showing 7 tools from fs") for line in first.screen()), (
            first.screen()
        )
    finally:
        await first.close()

    _ = (root / "notes.txt").write_text("hi", encoding="utf-8")
    second = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("listdir", {"path": "."}), text("there it is")]),
        journal=journal,
    )
    try:
        await second.type("what is here?")
        await second.until("listdir ok: 1 entries", "there it is")

        first, last = second.requests()
        assert "listdir" in _names(first), "a conversation that asked once does not ask again in the next process"
        assert _index(first) == (
            "6 more groups (15 tools): code(1), document(1), net(1), process(6), session(1), vcs(5) — "
            "use tool_require(group=…) to add one"
        ), "and the line it is given says the same thing it would have said before the restart"
        assert "file\tnotes.txt" in _told_of(last)
    finally:
        await second.close()


async def test_a_conversation_picked_out_of_the_journal_starts_a_tool_set_of_its_own(tmp_path, transport: str) -> None:
    """A conversation this process has just read back is one it has shown nothing to yet.

    The block the last process sent is not the block this one will send — the groups are the same and the
    way they are worded is not this process's to know — so the number a provider cache keys on moves as
    soon as the conversation is opened, and no request claims a prefix nobody has seen here (E176).
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root, transport=transport, model=ScriptedModel([text("first")]), journal=journal, tools=SILENT
    )
    try:
        await first.type("a thought")
        await first.finished()
        left = first.toolset_epoch
    finally:
        await first.close()

    second = await Story.start(
        root, transport=transport, model=ScriptedModel([text("second")]), journal=journal, tools=SILENT
    )
    try:
        await second.type("another thought")

        assert second.toolset_epoch == left + 1, "the journal's number is where this process starts counting"
        assert second.requests()[0].cache_key == f"main:{left + 1}:0", (
            "and the first request of the new process names a prefix of its own"
        )
    finally:
        await second.close()


async def test_a_conversation_the_journal_never_held_is_not_counted_against_it(tmp_path, transport: str) -> None:
    """The bump is about one conversation coming back, and one with nothing written about it starts at zero.

    A process that opens a conversation the journal has never seen has nothing to distrust: the block it
    sends is the block it built, and the number says so (E176).
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root, transport=transport, model=ScriptedModel([text("first")]), journal=journal, tools=SILENT
    )
    try:
        await first.type("a thought")
        await first.finished()
        assert first.toolset_epoch == 0, "nothing has changed its tool set yet"
    finally:
        await first.close()

    later = await Story.start(
        root, transport=transport, model=ScriptedModel([text("later")]), journal=journal, session="other", tools=SILENT
    )
    try:
        await later.type("a thought of my own")

        assert later.toolset_epoch == 0, "the other conversation left it nothing to count"
        assert later.requests()[0].cache_key == "other:0:0"
    finally:
        await later.close()
