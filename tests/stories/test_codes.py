"""One call that runs a program: what it may call, what it prints, and who is asked before it starts.

A program is the tool whose work is a loop — several calls, one answer, and only what it prints comes
back — so these stories are its outside: the question a person answers once for the plan, the text the
model is handed afterwards, and the three plans that are refused before anybody is asked anything.

The group is asked for first, because that is how a model reaches these tools at all: ``run_toolscript``
travels in ``code``, and nothing about it arrives before the round that asked (§10.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.cancel import CancelToken
from plyngent.kernel.trail import Effect
from plyngent.wire.ask import Severity
from tests.stories.model import calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: The first round of every story here: the group, and nothing else.
_GROUP = calling("tool_require", {"group": "code"})

#: A program that finds the notes and prints each one: a listing, a loop, and the one way back.
_FINDING = """\
found = glob_paths(pattern="*.txt")
for line in str(found).splitlines():
    print(line)
print("that is all")
"""


def _file(story: Story, name: str, said: str = "x") -> None:
    """A file in the story's workspace, put there before the story says anything."""
    _ = (story.workspace / name).write_text(said, encoding="utf-8")


async def test_a_program_calls_the_tools_in_its_plan_and_prints_what_it_found(story_with: Scripted) -> None:
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": _FINDING, "tools": ["glob_paths"], "intention": "find the notes"}),
        text("found them"),
    )
    _file(story, "notes.txt")

    await story.type("find the notes in one program")
    await story.answer("y")
    await story.finished()

    (question,) = story.questions()
    assert question.executable_code == _FINDING, "the person was shown the program that ran"
    assert question.severity is Severity.DANGER
    assert question.prompt == (
        "run a program that calls 1 tool (glob_paths)? a static check found no dunder access in it, "
        "and every call it makes comes back here"
    ), "and was told what it may call before being shown the text"

    answered = told(story.requests()[-1])
    assert "notes.txt" in answered, answered
    assert "that is all" in answered, "what the program printed is what came back"
    assert any(line.startswith("run_toolscript ok: the program printed") for line in story.screen()), story.screen()


async def test_a_result_a_program_never_prints_never_reaches_the_conversation(story_with: Scripted) -> None:
    """The whole of what a program buys: the calls it makes whose answers cost the conversation nothing."""
    code = 'print(len(str(glob_paths(pattern="*.txt"))))\n'
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["glob_paths"], "intention": "count the notes"}),
        text("counted"),
    )
    _file(story, "notes.txt")

    await story.type("count them")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1])
    assert "notes.txt" not in answered, "the listing the program read is not in the conversation"
    assert answered.startswith("the program printed"), answered


async def test_a_program_that_prints_nothing_says_so(story_with: Scripted) -> None:
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": "x = 1\n", "tools": ["glob_paths"], "intention": "think"}),
        text("nothing to say"),
    )

    await story.type("do something quiet")
    await story.answer("y")
    await story.finished()

    assert told(story.requests()[-1]) == "the program printed nothing, so nothing came back"


async def test_a_name_outside_the_plan_is_refused_before_anybody_is_asked(story_with: Scripted) -> None:
    story = await story_with(
        _GROUP,
        calling(
            "run_toolscript",
            {"code": 'print(call("read", path="notes.txt"))', "tools": ["glob_paths"], "intention": "read it"},
        ),
        text("oh"),
    )

    await story.type("read the note")
    await story.finished()

    refused = told(story.requests()[-1])
    assert "not in its plan" in refused, refused
    assert "glob_paths" in refused, "and the answer names what the program could have called"
    assert story.questions() == (), "a program that would be refused is nobody's question"


async def test_a_program_that_reaches_for_a_dunder_is_refused_and_not_run(story_with: Scripted) -> None:
    story = await story_with(
        _GROUP,
        calling(
            "run_toolscript", {"code": "print((1).__class__)", "tools": ["glob_paths"], "intention": "look inside"}
        ),
        text("oh"),
    )

    await story.type("reach for the class")
    await story.finished()

    assert any(
        line.startswith("run_toolscript refused: this program reaches for '__class__'") for line in story.screen()
    ), story.screen()
    assert story.questions() == (), "the scan answers this one, and a person is not put to it"


async def test_a_program_is_asked_about_every_time_because_what_was_agreed_was_this_text(
    story_with: Scripted,
) -> None:
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": 'print("one")', "tools": ["glob_paths"], "intention": "first"}),
        calling("run_toolscript", {"code": 'print("two")', "tools": ["glob_paths"], "intention": "second"}),
        text("both"),
    )

    await story.type("run two programs")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    asked = [question.executable_code for question in story.questions()]
    assert asked == ['print("one")', 'print("two")'], "a yes to one program is not a yes to the next one"


async def test_a_step_that_failed_ends_the_program_and_what_it_printed_still_arrives(story_with: Scripted) -> None:
    """A program's steps stand whether or not the program finishes: the text is its own account (E190)."""
    code = 'print("before")\nread(path="missing.txt")\nprint("after")\n'
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["read"], "intention": "read a note that is not there"}),
        text("oh"),
    )

    await story.type("read what is not there")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1])
    assert "before" in answered, "the line the program printed before the step that failed is still the answer"
    assert "after" not in answered, "and the program stopped where the step it did not catch failed"
    assert "missing.txt" in answered, answered
    assert any(line.startswith("run_toolscript failed:") for line in story.screen()), story.screen()


async def test_each_step_of_a_program_leaves_its_own_row_and_the_call_adds_no_version(story_with: Scripted) -> None:
    """A step has no result in the history to hang below, so its row is a root with its own address.

    The call that ran the program changes no path of its own, so the sweep behind it compares the tree
    with the version the last step left and finds nothing: two writes are two versions, and the third
    thing in the conversation is one text (E190, §10.12).
    """
    code = 'write_file(path="first.txt", content="one")\nwrite_file(path="second.txt", content="two")\nprint("both")\n'
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["write_file"], "intention": "write two notes"}),
        text("both"),
    )

    await story.type("write two notes in one program")
    await story.answer("y")
    await story.finished()

    rows = [row.payload for row in await story.host.trail.entries(story.session)]
    steps = [row for row in rows if isinstance(row, Effect)]
    assert [(row.call, row.tool, tuple(Path(one).name for one in row.changed)) for row in steps] == [
        ("tool-2/s1", "write_file", ("first.txt",)),
        ("tool-2/s2", "write_file", ("second.txt",)),
    ], "one row per step, under the call that ran the program, and nothing for the call itself"
    assert all(row.revision for row in steps), "and each of them names the version its bytes are kept under"

    store = story.host.history
    assert store is not None, "this host keeps versions, or there would be nothing to compare"
    head = await store.head(scope=CancelToken().at("story"))
    assert head == steps[-1].revision, "the version in force after the call is the one its last step left"


async def test_a_program_that_prints_past_its_budget_is_stopped_and_says_so(story_with: Scripted) -> None:
    """What a program prints is held in a buffer, so the buffer is where the size of it is counted."""
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": 'print("x" * 2_000_000)', "tools": ["glob_paths"], "intention": "flood"}),
        text("oh"),
    )

    await story.type("print more than a program may")
    await story.answer("y")
    await story.finished()

    assert told(story.requests()[-1]) == "the program printed more than 1048576 characters"
    assert any(line.startswith("run_toolscript failed: the program printed more than") for line in story.screen())


async def test_a_program_that_calls_past_its_budget_is_stopped_and_says_so(story_with: Scripted) -> None:
    """Two hundred steps is the wall: a loop that reads one file per line has no answer of its own."""
    code = 'for _ in range(205):\n    glob_paths(pattern="*.txt")\nprint("done")\n'
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["glob_paths"], "intention": "one call too many"}),
        text("oh"),
    )
    _file(story, "notes.txt")

    await story.type("walk past the wall")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1])
    assert "more than 200 calls" in answered, answered
    assert "done" not in answered, "the line after the budget ran is not in the answer"


async def test_a_result_can_be_read_as_a_value_and_its_handle_given_back(story_with: Scripted) -> None:
    """What a step answers is an object, so a program asks it questions instead of scraping its text."""
    code = """\
big = read(path="big.txt", limit=100000)
print(big.status, big.ok, big.failure == "", len(big.documents))
first = read(document_id=big.documents[0], limit=1)
print(str(first))
"""
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["read"], "intention": "read a big file in pieces"}),
        text("read it"),
    )
    _file(story, "big.txt", "needle\n" * 20_000)

    await story.type("read the big file")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1]).splitlines()
    assert "ok True True 1" in answered, answered
    assert "needle" in answered, "the handle the first step handed back is what the second one read"


async def test_a_step_that_would_ask_a_person_every_time_is_refused_inside_a_program(story_with: Scripted) -> None:
    """One question allowed the program, and a step cannot put another one in front of somebody (E190)."""
    code = 'write_file(path="notes.txt", content="new", overwrite=True)\nprint("wrote it")\n'
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["write_file"], "intention": "replace the note"}),
        text("oh"),
    )
    _file(story, "notes.txt", "old")

    await story.type("replace the note from inside a program")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1])
    assert "asks a person every time" in answered, answered
    assert "wrote it" not in answered, "the program stopped at the step that could not be agreed to"
    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "old", "and nothing was replaced"
    assert len(story.questions()) == 1, "the one question was about the program"


async def test_a_program_is_given_a_few_modules_and_no_way_to_reach_the_rest(story_with: Scripted) -> None:
    """The namespace is the fence: a module nobody gave it is refused, and the names that reach past are not there.

    The static scan says what it found and is evidence; this is what holds, which is why it is worth a story
    of its own (E190).
    """
    code = """\
import json
print("json:", json.dumps([1]))
try:
    import os
except Exception as exc:
    print("os:", exc)
try:
    open("notes.txt")
except NameError as exc:
    print("open:", exc)
try:
    getattr(json, "dumps")
except NameError as exc:
    print("getattr:", exc)
help("nope")
"""
    story = await story_with(
        _GROUP,
        calling("run_toolscript", {"code": code, "tools": ["read"], "intention": "ask what it was given"}),
        text("told"),
    )

    await story.type("ask the program what it can reach")
    await story.answer("y")
    await story.finished()

    answered = told(story.requests()[-1]).splitlines()
    assert "json: [1]" in answered, answered
    assert "os: import os" in answered, "a module it was not given is refused by the namespace's own import"
    assert any(line.startswith("open: name 'open' is not defined") for line in answered), answered
    assert any(line.startswith("getattr: name 'getattr' is not defined") for line in answered), answered
    assert "not here: nope. this program may call: read" in answered, "a name it cannot call lists the ones it can"
