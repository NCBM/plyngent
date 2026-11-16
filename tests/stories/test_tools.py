"""What a tool did, as the person watching and the model reading are told it.

A tool is one call and two readers: the terminal shows a line per call, and the model is handed the whole
report — the summary line, then whatever the tool produced to read, then any hint about what would change
the outcome. These stories are about that pair for the file and command tools: what the disk looks like
afterwards, what the model was told, and what a refusal names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from plyngent.kernel.model import Reply, Result
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from plyngent.kernel.model import Request

    type Scripted = Callable[..., Awaitable[Story]]


async def _told(story: Story) -> str:
    """All of what the model was read about the last call of the answer: the line, then the parts.

    The turn is waited for first: a tool hands its work to a thread — a directory walked, a program
    started, a file taken to the trash — and the request that carries the result comes after that.
    """
    await story.until("turn 1 finished")
    (_, second) = story.requests()
    said = second.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said.text


def _file(story: Story, name: str = "notes.txt", content: str = "hi") -> Path:
    """A file in the story's workspace, put there before the story says anything."""
    path = story.workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")
    return path


def _told_of(request: Request) -> str:
    """What the model was told at the end of one round: the result of the call that round answered."""
    said = request.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said.text


def _made(request: Request) -> Path:
    """The scratch directory the last thing the model was read named, read the way the model reads it."""
    found = re.search(r"temporary workspace: (.+)", _told_of(request))
    assert found is not None, f"nothing the model was read named a directory: {_told_of(request)!r}"
    return Path(found.group(1).strip())


async def test_a_listing_shows_what_is_in_a_directory(story_with: Scripted) -> None:
    story = await story_with(calling("listdir", {"path": "."}), text("that is all"))
    _ = _file(story)
    _ = (story.workspace / "sub").mkdir()

    await story.require("fs")

    await story.type("what is here?")

    told = await _told(story)
    assert "file\tnotes.txt" in told, told
    assert "dir\tsub" in told, "a directory says so in its own column, so the model can tell them apart"
    assert story.questions() == (), "reading changes nothing, so nothing is asked"


async def test_a_listing_says_the_columns_it_was_asked_for(story_with: Scripted) -> None:
    story = await story_with(
        calling("listdir", {"path": ".", "stat": ["mode", "size", "symlink"]}), text("that is all")
    )
    _ = _file(story, "notes.txt", "12345")
    _ = (story.workspace / "link").symlink_to(story.workspace / "notes.txt")

    await story.require("fs")

    await story.type("what is here, with the details?")

    told = await _told(story)
    assert told.startswith("2 entries"), told
    rows = {line.split("\t")[1]: line.split("\t")[2:] for line in told.splitlines()[1:]}
    assert len(rows["notes.txt"][0]) == 3, "the permission bits, octal"
    assert rows["notes.txt"][1] == "5", "size in bytes"
    assert rows["notes.txt"][2] == "-", "a file that is not a symlink has nothing to point at"
    assert rows["link"][2] == str((story.workspace / "notes.txt").resolve()), "a symlink says where it points"


async def test_a_listing_of_something_that_is_not_a_directory_is_the_worlds_answer(story_with: Scripted) -> None:
    story = await story_with(calling("listdir", {"path": "notes.txt"}), text("fair"))
    _ = _file(story)

    await story.require("fs")

    await story.type("list that file")

    assert "listdir failed: not a directory: notes.txt" in story.screen(), (
        "the model asked about the wrong kind of thing, and the answer says which kind it is"
    )


async def test_a_tree_shows_the_shape_of_a_directory(story_with: Scripted) -> None:
    story = await story_with(calling("tree", {"path": ".", "max_depth": 2}), text("that is all"))
    _ = _file(story, "sub/deep.txt", "x")

    await story.require("fs")

    await story.type("what is the shape of this?")

    told = await _told(story)
    assert "sub/" in told and "deep.txt" in told, told


async def test_a_tree_can_be_drawn_three_ways(story_with: Scripted) -> None:
    story = await story_with(
        calling("tree", {}),
        calling("tree", {"format": "flat"}),
        calling("tree", {"format": "decorated"}),
        text("done"),
    )
    _ = _file(story, "b.txt", "x")
    _ = _file(story, "src/a.py", "x")

    await story.require("fs")

    await story.type("how is this shaped?")
    await story.finished()

    (_, bullets, flat, boxes) = story.requests()
    assert _told_of(bullets).splitlines() == ["3 entries", "- b.txt", "- src/", "  - a.py"], (
        "a nested bullet list, two spaces per level"
    )
    assert _told_of(flat).splitlines() == ["3 entries", "b.txt", "src/", "src/a.py"], (
        "every path usable as it is, because a flat list has no levels to infer"
    )
    assert _told_of(boxes).splitlines() == ["3 entries", "./", "├── b.txt", "└── src/", "    └── a.py"], (
        "directories and files in name order, with the branch that says which is last"
    )


async def test_a_tree_budget_is_per_directory(story_with: Scripted) -> None:
    story = await story_with(
        calling("tree", {"max_entries": 2}),
        calling("tree", {"format": "flat", "max_entries": 2}),
        text("that is all"),
    )
    _ = _file(story, "a.txt", "x")
    for name in ("1.txt", "2.txt", "3.txt"):
        _ = _file(story, f"big/{name}", "x")

    await story.require("fs")

    await story.type("the first couple of each")
    await story.finished()

    (_, bullets, flat) = story.requests()
    assert _told_of(bullets).splitlines() == [
        "4 of 5 entries",
        "- a.txt",
        "- big/",
        "  - 1.txt",
        "  - 2.txt",
        "… (1 more entries not shown)",
    ], "one big directory cannot hide the rest, because each directory gives up its own budget"
    assert _told_of(flat).splitlines() == ["2 of 5 entries", "a.txt", "big/", "… (3 more entries not shown)"], (
        "a flat list has no levels to bound, so there the budget is over the lines"
    )


async def test_a_glob_finds_by_pattern(story_with: Scripted) -> None:
    story = await story_with(calling("glob_paths", {"pattern": "**/*.txt"}), text("that is all"))
    _ = _file(story, "notes.txt")
    _ = _file(story, "sub/deep.txt")
    _ = _file(story, "sub/skip.md")

    await story.type("which text files are here?")

    told = await _told(story)
    assert "notes.txt" in told and "deep.txt" in told, told
    assert "skip.md" not in told, "the pattern is what it asked for"


async def test_a_glob_leaves_out_what_it_was_not_asked_for_and_the_filter_is_how_to_see_it(
    story_with: Scripted,
) -> None:
    story = await story_with(
        calling("glob_paths", {"pattern": "**/*.py"}),
        calling("glob_paths", {"pattern": "**/*.py", "filter": {"show_hidden": True}}),
        text("done"),
    )
    _ = _file(story, "src/a.py", "x")
    _ = _file(story, ".git/b.py", "x")
    _ = _file(story, ".hidden.py", "x")

    await story.type("which python files are here?")
    await story.finished()

    (_, quiet, loud) = story.requests()
    assert _told_of(quiet).splitlines() == ["2 paths", ".hidden.py", "src/a.py"], _told_of(quiet)
    assert _told_of(loud).splitlines() == ["3 paths", ".git/b.py", ".hidden.py", "src/a.py"], (
        "a dot directory is as hidden as a dot file, so one switch shows both"
    )


async def test_a_glob_stops_at_its_budget_and_says_so(story_with: Scripted) -> None:
    story = await story_with(calling("glob_paths", {"pattern": "*.py", "max_matches": 2}), text("that is all"))
    for name in ("a.py", "b.py", "c.py"):
        _ = _file(story, name, "x")

    await story.type("the first two")

    assert (await _told(story)).splitlines() == ["2 paths", "a.py", "b.py", "...[truncated at 2 matches]"], (
        "a budget that cut the list says so, so the model knows there is more"
    )


async def test_a_copy_and_a_move_change_the_tree(story_with: Scripted) -> None:
    story = await story_with(
        calling("copy_path", {"src": "notes.txt", "dst": "copy.txt"}),
        calling("move_path", {"src": "copy.txt", "dst": "renamed.txt", "intention": "give it a better name"}),
        text("done"),
    )
    _ = _file(story)

    await story.require("fs")

    await story.type("copy it, then rename the copy")
    await story.answer("y")
    await story.answer("y")

    assert (story.workspace / "notes.txt").read_text(encoding="utf-8") == "hi"
    assert (story.workspace / "renamed.txt").read_text(encoding="utf-8") == "hi"
    assert not (story.workspace / "copy.txt").exists()
    assert "copy_path ok: copied file notes.txt -> copy.txt" in story.screen(), "the person is told what each call did"
    assert "move_path ok: moved file copy.txt -> renamed.txt" in story.screen()


async def test_a_delete_takes_the_file_away(story_with: Scripted) -> None:
    story = await story_with(
        calling("delete_path", {"path": "notes.txt", "intention": "it is stale"}),
        text("gone"),
    )
    _ = _file(story)

    await story.require("fs")

    await story.type("delete the note")
    await story.answer("y")

    assert not (story.workspace / "notes.txt").exists(), "a delete deletes"
    assert "deleted file notes.txt" in await _told(story), "the model is told what left the tree"


async def test_a_call_that_leaves_nothing_behind_has_to_say_why(story_with: Scripted) -> None:
    story = await story_with(calling("delete_path", {"path": "notes.txt"}), text("fair"))
    _ = _file(story)

    await story.require("fs")

    await story.type("delete the note")

    assert (story.workspace / "notes.txt").exists(), "a call that was refused is a call that did nothing"
    assert story.questions() == (), "it never got as far as asking anybody"
    assert any(line.startswith("delete_path failed:") and "intention" in line for line in story.screen()), (
        story.screen()
    )
    told = await _told(story)
    assert "intention" in told, "the model is told which field it left out, so the next call can carry it"


async def test_a_program_runs_and_both_commands_are_reported(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["sh", "-c", "echo one"]}, {"argv": ["sh", "-c", "echo two"]}],
                "intention": "say one and two",
            },
        ),
        text("both ran"),
    )

    await story.type("say one and two")

    told = await _told(story)
    assert "one" in told and "two" in told, told
    assert "exit_code=0" in told, "the exit code is part of the record, not a verdict"


async def test_a_program_that_fails_is_a_result_and_not_an_error(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {"steps": [{"argv": ["sh", "-c", "echo no; exit 3"]}], "intention": "a program that fails"},
        ),
        text("oh"),
    )

    await story.type("run something that fails")

    told = await _told(story)
    assert "exit_code=3" in told, told
    assert "no" in told, "what it said on the way out is kept"


async def test_a_programs_input_can_be_a_lump_of_text(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {
                "steps": [{"argv": ["sh", "-c", "cat"], "stdin": {"type": "text", "text": "through stdin"}}],
                "intention": "read its input back",
            },
        ),
        text("read it back"),
    )

    await story.type("feed it a line")

    assert "through stdin" in await _told(story)


async def test_a_program_that_is_not_there_fails_the_call(story_with: Scripted) -> None:
    story = await story_with(
        calling(
            "execute",
            {"steps": [{"argv": ["no-such-program"]}], "intention": "run something missing"},
        ),
        text("oh"),
    )

    await story.type("run something missing")

    told = await _told(story)
    assert "no-such-program" in told, told


async def test_a_path_outside_the_workspace_is_refused_and_says_where_the_tree_ends(story_with: Scripted) -> None:
    story = await story_with(calling("listdir", {"path": "../elsewhere"}), text("oh"))

    await story.require("fs")

    await story.type("look outside")

    told = await _told(story)
    assert told.startswith("path escapes workspace root"), told
    assert "../elsewhere" in told, "the refusal quotes what was asked for"
    assert "hint: request_directory_access" in told, (
        "the model is handed the step that would work, as the call to make — not as a sentence about it"
    )
    assert "hint: new_temporary_workspace" in told, "and the other way through, since there are two"


async def test_a_granted_path_reads_again_and_still_reads_after_a_restart(tmp_path: Path, transport: str) -> None:
    """Acceptance ①: the refusal, the grant it names, and the same call working — here and in the next process."""
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    _ = (outside / "notes.txt").write_text("kept outside", encoding="utf-8")
    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel(
            [
                calling("listdir", {"path": "../elsewhere"}),
                calling(
                    "request_directory_access",
                    {"path": "../elsewhere", "mode": "read", "intention": "the notes are kept outside"},
                ),
                calling("listdir", {"path": "../elsewhere"}),
                text("it is readable now"),
            ]
        ),
        journal=journal,
    )
    try:
        await first.require("fs")
        await first.type("look outside")
        await first.answer("y")
        await first.finished()

        assert any(line.startswith("request_directory_access ok: allowed read") for line in first.screen()), (
            first.screen()
        )
        assert any(line.startswith("listdir ok: 1 entries") for line in first.screen()), (
            "the same call that was refused now reads the directory a person allowed"
        )
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("listdir", {"path": "../elsewhere"}), text("still readable")]),
        journal=journal,
    )
    try:
        await second.require("fs")
        await second.type("is it still there?")

        assert any(line.startswith("listdir ok: 1 entries") for line in second.screen()), (
            "a grant belongs to the conversation, so it is still there in the next process"
        )
    finally:
        await second.close()


async def test_a_temporary_workspace_is_written_in_and_gone_when_the_host_stops(story_with: Scripted) -> None:
    """Acceptance ②: the other way to lift a refusal — a directory the host makes, and later removes."""
    made: list[Path] = []

    def scratch(request: Request) -> Reply:
        """The write a model would make next: inside the directory it was just handed."""
        made.append(_made(request))
        return calling("write_file", {"path": str(made[-1] / "notes.txt"), "content": "hi"})

    story = await story_with(
        calling("write_file", {"path": "../elsewhere/notes.txt", "content": "hi"}),
        calling("new_temporary_workspace", {"prefix": "story"}),
        scratch,
        text("written"),
    )

    await story.require("fs")
    await story.type("write outside the workspace")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    assert any("new_temporary_workspace ok: temporary workspace: " in line for line in story.screen())
    assert (made[0] / "notes.txt").read_text(encoding="utf-8") == "hi", "a path under it needs no grant"
    assert await story.stop() == 0

    await story.close()
    assert not made[0].exists(), "the scratch directory goes when the host does"


async def test_a_write_makes_the_directories_it_needs(story_with: Scripted) -> None:
    story = await story_with(calling("write_file", {"path": "a/b/c.txt", "content": "hi"}), text("done"))

    await story.type("write it deep")
    await story.answer("y")

    assert (await _told(story)).startswith("wrote 2 chars to a/b/c.txt"), "the line names what was written, and where"
    assert (story.workspace / "a" / "b" / "c.txt").read_text(encoding="utf-8") == "hi"


async def test_replacing_a_file_is_refused_with_the_one_value_that_allows_it(story_with: Scripted) -> None:
    story = await story_with(
        calling("write_file", {"path": "notes.txt", "content": "new"}),
        calling("write_file", {"path": "notes.txt", "content": "new", "overwrite": True}),
        text("done"),
    )
    existing = _file(story, "notes.txt", "old")

    await story.type("replace the note")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    (_, refused, replaced) = story.requests()
    assert _told_of(refused) == ("file exists: notes.txt\nhint: write_file(overwrite=true) — replace it on purpose"), (
        "the refusal carries the call that would have worked, not a sentence about one"
    )
    assert existing.read_text(encoding="utf-8") == "new"
    assert _told_of(replaced).startswith("wrote 3 chars to notes.txt")
    assert [question.prompt for question in story.questions()] == ["allow write_file on notes.txt?"] * 2, (
        "the second call replaces a file that was already there, and a replacement is not an ordinary write"
    )


async def test_a_copy_takes_a_whole_directory_and_will_not_land_on_one_unless_told(story_with: Scripted) -> None:
    story = await story_with(
        calling("copy_path", {"src": "src", "dst": "dst"}),
        calling("copy_path", {"src": "src", "dst": "dst"}),
        calling("copy_path", {"src": "src", "dst": "dst", "overwrite": True}),
        text("done"),
    )
    _ = _file(story, "src/a.txt", "x")

    await story.require("fs")

    await story.type("copy the tree")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    (_, copied, refused, replaced) = story.requests()
    assert _told_of(copied) == "copied directory src -> dst"
    assert _told_of(refused) == (
        "destination exists: dst\nhint: copy_path(overwrite=true) — replace what is already there"
    )
    assert _told_of(replaced).startswith("copied directory src -> dst")
    assert (story.workspace / "dst" / "a.txt").read_text(encoding="utf-8") == "x"


async def test_a_file_copied_onto_a_directory_lands_inside_it(story_with: Scripted) -> None:
    story = await story_with(calling("copy_path", {"src": "a.txt", "dst": "sub"}), text("done"))
    _ = _file(story, "a.txt", "x")
    _ = (story.workspace / "sub").mkdir()

    await story.require("fs")

    await story.type("put a copy of it in there")
    await story.answer("y")

    assert (await _told(story)) == "copied file a.txt -> sub/a.txt", "a directory as a destination means what it means"
    assert (story.workspace / "sub" / "a.txt").read_text(encoding="utf-8") == "x"


async def test_a_move_into_itself_is_the_arguments_and_not_the_rules(story_with: Scripted) -> None:
    story = await story_with(
        calling("move_path", {"src": "src", "dst": "src/nested", "intention": "tidy it up"}),
        text("fair"),
    )
    _ = _file(story, "src/a.txt", "x")

    await story.require("fs")

    await story.type("move it inside itself")
    await story.answer("y")
    await story.finished()

    assert _told_of(story.requests()[-1]) == "cannot move src inside itself", "nothing moved, and the arguments are why"


async def test_taking_away_a_directory_has_to_be_asked_for_and_the_root_never_is(story_with: Scripted) -> None:
    story = await story_with(
        calling("delete_path", {"path": "sub", "intention": "clear it out"}),
        calling("delete_path", {"path": "sub", "recursive": True, "intention": "clear it out for real"}),
        calling("delete_path", {"path": ".", "recursive": True, "intention": "start over"}),
        text("done"),
    )
    _ = _file(story, "sub/a.txt", "x")

    await story.require("fs")

    await story.type("clear the scratch directory")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    (_, refused, gone, root) = story.requests()
    assert _told_of(refused) == "directory not empty: sub\nhint: delete_path(recursive=true) — remove it whole", (
        "a directory is not taken away by accident"
    )
    assert not (story.workspace / "sub").exists()
    assert _told_of(gone).startswith("deleted directory sub")
    assert _told_of(root) == "cannot delete the workspace root", "no grant lifts this one, so there is nothing to offer"
    assert story.workspace.exists()
