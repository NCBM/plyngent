"""One call that runs a program: what it may call, what it prints, and who is asked before it starts.

A program is the tool whose work is a loop — several calls, one answer, and only what it prints comes
back — so these stories are its outside: the question a person answers once for the plan, the text the
model is handed afterwards, and the three plans that are refused before anybody is asked anything.

The group is asked for first, because that is how a model reaches these tools at all: ``run_toolscript``
travels in ``code``, and nothing about it arrives before the round that asked (§10.4).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
