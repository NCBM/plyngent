"""One answer that asks for several calls: what may overlap, and what the history says came of them.

A model may answer with a batch — several calls at once — and which of them could have been run at the
same time is not the model's business. What is, and what these stories read, is that the calls which
depend on each other see each other's work: a call that reads what an earlier call in the same answer
wrote reads the write, and a command waits for the file work of its own answer before it starts.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from plyngent.kernel.model import Call, Reply
from plyngent_creative.model import text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


def _asked(tool: str, **arguments: object) -> Call:
    """One call of a batch: the model's own id for it, and its arguments as the model would spell them."""
    return Call(id=tool, tool=tool, arguments=json.dumps(arguments))


def _batch(*calls: Call) -> Reply:
    """One answer that asks for several calls, in the order the model wrote them."""
    return Reply(calls=calls)


async def test_a_call_reads_what_the_call_before_it_wrote(story_with: Scripted) -> None:
    story = await story_with(
        _batch(
            _asked("write_file", path="notes.txt", content="hi"), _asked("copy_path", src="notes.txt", dst="back.txt")
        ),
        text("copied it back"),
    )

    await story.require("fs")

    await story.type("write the note and copy it")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi"
    assert (story.workspace / "back.txt").read_text(encoding="utf-8") == "hi", (
        "the copy read the note the same answer had just written, so it read what it wrote"
    )
    (_, second) = story.requests()
    assert [message.tool for message in second.messages[-2:]] == ["write_file", "copy_path"], (
        "the results are in the order the model asked for them, whichever call came back first"
    )


async def test_a_command_reads_the_file_another_command_wrote_in_the_same_answer(story_with: Scripted) -> None:
    """Neither call names its path as a field: a redirect and an input file are inside the tree.

    That is the case the walk exists for. ``roles`` names one argument per path, and a tree of commands
    has nothing to name, so before ``execute`` said where its own paths are the pair would have started at
    once and the read would have found whatever was on disk a moment earlier.
    """
    story = await story_with(
        _batch(
            _asked("execute", steps=[{"argv": ["echo", "hi"], "pipe": "out.txt"}], intention="write the note"),
            _asked(
                "execute",
                steps=[{"argv": ["cat"], "stdin": {"type": "file", "path": "out.txt"}}],
                intention="and read it back",
            ),
        ),
        text("done"),
    )

    await story.type("write it and read it")
    await story.finished()

    (_, second) = story.requests()
    read = second.messages[-1]
    assert read.tool == "execute"
    assert "hi" in read.text, "the read waited for the write, because both calls named that one file"


async def test_a_command_sees_the_files_its_own_answer_wrote(story_with: Scripted) -> None:
    story = await story_with(
        _batch(
            _asked("write_file", path="notes.txt", content="hi"),
            _asked("execute", steps=[{"argv": ["cat", "notes.txt"]}], intention="read the note back"),
        ),
        text("done"),
    )

    await story.type("write the note and print it")
    await story.answer("y")
    await story.finished()

    (_, second) = story.requests()
    ran = second.messages[-1]
    assert ran.tool == "execute"
    assert "hi" in ran.text, "a command waits for its own answer's file work, so it found what was written"
