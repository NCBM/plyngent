"""What the document tools do to one operand, and what the model is told about it.

The document group asks one question of one thing — what is in it — and that thing is named either by
a ``path`` or by a ``document_id``. These stories are the reading half: the window a call gets back,
the shape of a numbered read, and the answers that are not a window at all.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.documents import MAX_TEXT_CHARS
from plyngent.kernel.model import Reply, Result
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, Step, calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from plyngent.kernel.model import Request

    type Scripted = Callable[..., Awaitable[Story]]


async def _told(story: Story) -> str:
    """Wait for the turn to end, then read what the model was given about the last call."""
    await story.until("turn 1 finished")
    return told(story.requests()[-1])


def _told_of(request: Request) -> str:
    """What the model was told at the end of one round: the result of the call that round answered."""
    said = request.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said.text


def _file(story: Story, name: str = "notes.txt", content: str = "hi") -> Path:
    """A file in the story's workspace, put there before the story says anything."""
    path = story.workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(content, encoding="utf-8")
    return path


async def test_a_read_gives_the_whole_file_by_default(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "notes.txt"}), text("that is all"))
    _ = _file(story, "notes.txt", "one\ntwo\n")

    await story.type("what does the note say?")

    assert await _told(story) == "read notes.txt (lines 1-2 of 2)\none\ntwo\n", "the line, then what is in it"
    assert story.questions() == (), "reading changes nothing, so nothing is asked"


async def test_a_read_takes_the_window_it_was_asked_for(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "notes.txt", "offset": 1, "limit": 2}), text("that is all"))
    _ = _file(story, "notes.txt", "one\ntwo\nthree\nfour\n")

    await story.type("the middle of it")

    assert await _told(story) == "read notes.txt (lines 2-3 of 4)\ntwo\nthree\n", (
        "offset counts from zero, and the summary says which lines that came to"
    )


async def test_a_numbered_read_numbers_the_lines_it_shows(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "offset": 1, "limit": 2, "with_lineno": True}), text("that is all")
    )
    _ = _file(story, "notes.txt", "one\ntwo\nthree\nfour\n")

    await story.type("with the numbers, please")

    assert (await _told(story)).splitlines() == ["read notes.txt (lines 2-3 of 4)", "     2|two", "     3|three"], (
        "a fixed-width column, numbered from one, in the shape edit_lineno counts by"
    )


async def test_a_read_of_an_empty_file_says_it_is_empty(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "empty.txt"}), text("fair"))
    _ = _file(story, "empty.txt", "")

    await story.type("read the empty one")

    assert await _told(story) == "read empty.txt (nothing in it)", "nothing is not the same as nothing found"


async def test_a_read_past_the_end_says_how_many_lines_there_are(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "notes.txt", "offset": 9}), text("fair"))
    _ = _file(story, "notes.txt", "one\ntwo\n")

    await story.type("read from way down")

    assert await _told(story) == "read notes.txt: no lines at offset 9 (it has 2, offset is 0-based)", (
        "an empty answer still says what the call was up against"
    )


async def test_a_read_of_what_is_not_there_names_what_was_asked_for(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "nope.txt"}), text("fair"))

    await story.type("read the missing one")

    assert await _told(story) == "file not found: nope.txt", "the world's answer, with the name as it was given"


async def test_a_read_of_a_directory_is_told_which_tool_lists_one(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "sub"}), text("oh"))
    _ = (story.workspace / "sub").mkdir()

    await story.type("read the directory")

    assert (await _told(story)).splitlines() == [
        "not a file: sub",
        "hint: tool_require(group=fs) — see what is in it",
    ], "the step that would work is in a group this session was never shown, so the hint says how to get it"


async def test_a_read_of_bytes_that_are_not_utf8_says_what_it_cost(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "latin.txt"}), text("oh"))
    _ = (story.workspace / "latin.txt").write_bytes(b"caf\xe9\n")

    await story.type("read the odd one")

    told = await _told(story)
    assert told.startswith("read latin.txt (lines 1-1 of 1) — not utf-8"), told
    assert "\ufffd" in told, "what could not be read is a replacement character, and the summary says so"


async def test_a_read_has_to_name_exactly_one_thing_to_read(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "document_id": "d1"}),
        calling("read", {"offset": 2}),
        text("done"),
    )
    _ = _file(story)

    await story.type("read it two ways at once")
    await story.finished()

    (_, both, neither) = story.requests()
    assert _told_of(both).startswith("give either 'path' or 'document_id', not both"), "two locators is a guess"
    assert _told_of(neither).startswith("give 'path' (a file in the workspace) or 'document_id'"), (
        "and no locator is not a request"
    )


async def test_an_edit_changes_the_first_one_and_says_what_is_left(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_replace", {"path": "notes.txt", "old_string": "a", "new_string": "x"}), text("done")
    )
    note = _file(story, "notes.txt", "a\nb\na\n")

    await story.type("change the first a")
    await story.answer("y")

    assert note.read_text(encoding="utf-8") == "x\nb\na\n", "one occurrence, left to right"
    assert (await _told(story)).splitlines() == [
        "replaced 1 occurrence in notes.txt ('a' → 'x') (1 of 2 matches)",
        "hint: edit_replace(max_replaces=2) — 1 more of them are still there",
    ], "what was replaced, out of how many, and the call that would do the rest"


async def test_an_edit_takes_a_budget_and_one_that_covers_everything_says_nothing_more(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_replace", {"path": "notes.txt", "old_string": "a", "new_string": "x", "max_replaces": 2}),
        text("done"),
    )
    note = _file(story, "notes.txt", "a\nb\na\n")

    await story.type("change both")
    await story.answer("y")

    assert note.read_text(encoding="utf-8") == "x\nb\nx\n"
    assert await _told(story) == "replaced 2 occurrences in notes.txt ('a' → 'x')", (
        "nothing is left, so nothing is offered"
    )


async def test_an_edit_of_what_is_not_in_the_file_points_at_reading_it(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_replace", {"path": "notes.txt", "old_string": "zzz", "new_string": "x"}), text("oh")
    )
    note = _file(story, "notes.txt", "a\n")

    await story.type("change something that is not there")
    await story.answer("y")

    assert note.read_text(encoding="utf-8") == "a\n", "a call that found nothing changed nothing"
    assert (await _told(story)).splitlines() == [
        "old_string not found in notes.txt",
        "hint: read(path=notes.txt) — see what is there now",
    ], "the step that would show what to replace is offered as a call"


async def test_a_numbered_edit_changes_the_lines_a_numbered_read_showed(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 2, "end_line": 2, "new_content": "TWO\n"}),
        text("done"),
    )
    note = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("change the middle line")
    await story.answer("y")
    await story.finished()

    assert note.read_text(encoding="utf-8") == "one\nTWO\nthree\n", (
        "the numbers a read wrote are the numbers an edit takes"
    )
    assert _told_of(story.requests()[-1]) == "replaced line 2 (1 line; first: 'two') with 1 line in notes.txt"


async def test_a_numbered_edit_appends_after_the_last_line(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 4, "end_line": 4, "new_content": "four\n"}),
        text("done"),
    )
    note = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("add a line at the end")
    await story.answer("y")
    await story.finished()

    assert note.read_text(encoding="utf-8") == "one\ntwo\nthree\nfour\n"
    assert _told_of(story.requests()[-1]) == "appended 1 line after line 3 in notes.txt"


async def test_an_append_into_an_empty_file_needs_no_read(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_lineno", {"path": "new.txt", "start_line": 1, "end_line": 1, "new_content": "hello\n"}),
        text("done"),
    )
    note = _file(story, "new.txt", "")

    await story.type("start it off")
    await story.answer("y")
    await story.finished()

    assert note.read_text(encoding="utf-8") == "hello\n", "there were no lines to clobber, so there was nothing to read"
    assert _told_of(story.requests()[-1]) == "appended 1 line to new.txt"


async def test_a_numbered_edit_will_not_touch_lines_that_were_not_read(story_with: Scripted) -> None:
    story = await story_with(
        calling("edit_lineno", {"path": "notes.txt", "start_line": 2, "end_line": 2, "new_content": "TWO\n"}),
        text("fair"),
    )
    note = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("change a line nobody read")
    await story.answer("y")

    assert note.read_text(encoding="utf-8") == "one\ntwo\nthree\n", "nothing was changed"
    assert (await _told(story)).splitlines() == [
        "notes.txt: line 2 not read with numbers",
        "hint: read(path=notes.txt, offset=0, limit=3, with_lineno=true) — read it before changing it",
    ], "the numbered read covering the range, with a margin around it"


async def test_a_numbered_edit_of_a_file_that_changed_since_it_was_read_is_refused(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("write_file", {"path": "notes.txt", "content": "one\ntwo\nthree\n", "overwrite": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 2, "end_line": 2, "new_content": "TWO\n"}),
        text("done"),
    )
    note = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("read it, rewrite it, then edit by the old numbers")
    await story.answer("y")
    await story.answer("y")
    await story.finished()

    assert (await _told(story)).splitlines() == [
        "notes.txt changed since it was read: its line numbers may be stale",
        "hint: read(path=notes.txt, offset=0, limit=3, with_lineno=true) — read it before changing it",
    ]
    assert note.read_text(encoding="utf-8") == "one\ntwo\nthree\n", "a stale range is not guessed at"


async def test_a_numbered_edit_keeps_the_line_endings_the_file_had(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 2, "end_line": 2, "new_content": "X\n"}),
        text("done"),
    )
    note = _file(story, "notes.txt", "one\r\ntwo\r\nthree\r\n")

    await story.type("change the middle line of a file with carriage returns")
    await story.answer("y")

    assert note.read_bytes() == b"one\r\nX\r\nthree\r\n", "the model wrote newlines; the file's own ending wins"


async def test_an_empty_new_content_takes_the_range_away(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 2, "end_line": 2, "new_content": ""}),
        text("done"),
    )
    note = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("drop the middle line")
    await story.answer("y")
    await story.finished()

    assert note.read_text(encoding="utf-8") == "one\nthree\n"
    assert _told_of(story.requests()[-1]) == "replaced line 2 (1 line; first: 'two') with 0 lines in notes.txt"


async def test_a_range_past_the_end_is_the_arguments(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "notes.txt", "with_lineno": True}),
        calling("edit_lineno", {"path": "notes.txt", "start_line": 9, "end_line": 9, "new_content": "x\n"}),
        text("fair"),
    )
    _ = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.type("edit a line that does not exist")
    await story.answer("y")
    await story.finished()

    assert _told_of(story.requests()[-1]) == "start_line 9 is past the end of the file (3 lines)", (
        "the file says how long it is, so the next number is the model's to choose"
    )


async def test_a_body_too_long_to_send_is_kept_and_the_result_carries_the_handle(story_with: Scripted) -> None:
    content = _log()
    story = await story_with(calling("read", {"path": "big.log"}), text("that is a lot"))
    _ = _file(story, "big.log", content)

    await story.type("read the whole log")

    told = await _told(story)
    lines = len(content.splitlines())
    assert told.splitlines() == [
        f"read big.log (lines 1-{lines} of {lines})",
        f"[d1] big.log, lines 1-{lines} of {lines} ({len(content)} chars)",
    ], "what the model gets is a handle and the line that stands in for what it cannot hold"
    assert f"document d1 kept ({len(content)} chars)" in story.screen(), (
        "the person watching is told that what they are not seeing still exists"
    )


async def test_a_kept_document_is_read_back_by_its_handle(story_with: Scripted) -> None:
    content = _log()
    story = await story_with(
        calling("read", {"path": "big.log"}),
        _by_handle("read", {"offset": 0, "limit": 2}),
        text("that is enough"),
    )
    _ = _file(story, "big.log", content)

    await story.type("read the log, then the first couple of lines of it")
    await story.finished()

    assert _told_of(story.requests()[-1]) == "read d1 (lines 1-2 of 4000)\nline 0\nline 1\n", (
        "the handle is read as a window, the same shape a file is read as"
    )


async def test_a_handle_this_session_does_not_have_is_answered_with_the_list(story_with: Scripted) -> None:
    """A handle that names nothing — made up, or evicted when the store made room — has one answer (§0.6, R10)."""
    story = await story_with(calling("read", {"document_id": "d99", "limit": 1}), text("fair"))

    await story.require("document")

    await story.type("read the thing from before")

    assert (await _told(story)).splitlines() == [
        "no document d99 in this session",
        "hint: documents — see what was kept",
    ], "the fix is the list, because the list is what says which handles this session has"


async def test_documents_says_what_this_conversation_kept(story_with: Scripted) -> None:
    content = _log()
    story = await story_with(
        calling("read", {"path": "big.log"}),
        calling("documents", {}),
        text("noted"),
    )
    _ = _file(story, "big.log", content)

    await story.type("read the log, then ask what was kept")
    await story.finished()

    lines = len(content.splitlines())
    assert _told_of(story.requests()[-1]).splitlines() == [
        "1 kept",
        f"d1\ttext/plain\t{len(content)} chars\tbig.log, lines 1-{lines} of {lines}",
    ], "the handle, what it is, how big it is, and which lines of which file it came from"


async def test_a_conversation_that_kept_nothing_has_no_documents(story_with: Scripted) -> None:
    story = await story_with(calling("documents", {}), text("noted"))

    await story.type("what have you kept?")

    assert await _told(story) == "nothing was kept in this conversation", (
        "an empty list is a fact about the conversation, not a failure of the question"
    )


async def test_a_handle_outlives_the_process_that_made_it(tmp_path, transport: str) -> None:
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    content = _log()

    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("read", {"path": "big.log"}), text("kept")]),
        journal=journal,
    )
    try:
        _ = (root / "big.log").write_text(content, encoding="utf-8")
        await first.type("read the log")
        assert any("document d1 kept" in line for line in first.screen()), first.screen()
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("read", {"document_id": "d1", "limit": 1}), text("there it is")]),
        journal=journal,
    )
    try:
        await second.type("the first line of the log I kept")
        await second.finished(2)  # the first turn of this journal was the one that made the handle

        assert _told_of(second.requests()[-1]) == "read d1 (lines 1-1 of 4000)\nline 0\n", (
            "a handle names content, and the content is still there after a restart"
        )
        assert (tmp_path / "documents.db").exists(), "kept beside the journal, so one move takes both"
    finally:
        await second.close()


async def test_a_search_finds_the_lines_that_match(story_with: Scripted) -> None:
    story = await story_with(calling("search", {"path": "notes.txt", "pattern": "two"}), text("found it"))
    _ = _file(story, "notes.txt", "one\ntwo\nthree\n")

    await story.require("document")

    await story.type("which line says two?")

    assert await _told(story) == "1 match in notes.txt\n2: two", (
        "the line number the model is shown is the number edit_lineno counts by"
    )


async def test_a_search_with_no_hits_says_so_about_what_it_looked_in(story_with: Scripted) -> None:
    story = await story_with(calling("search", {"path": "notes.txt", "pattern": "zzz"}), text("fair"))
    _ = _file(story, "notes.txt", "one\ntwo\n")

    await story.require("document")

    await story.type("look for something that is not there")

    assert await _told(story) == "no line of notes.txt matches 'zzz'", "no hits is an answer, not a failure"


async def test_a_search_stops_at_its_budget_and_says_what_it_left(story_with: Scripted) -> None:
    story = await story_with(calling("search", {"path": "notes.txt", "pattern": "x", "max_matches": 2}), text("enough"))
    _ = _file(story, "notes.txt", "x\nx\nx\n")

    await story.require("document")

    await story.type("the first two")

    assert (await _told(story)).splitlines() == [
        "2 of 3 matches in notes.txt",
        "1: x",
        "2: x",
        "...[truncated at 2 matches]",
    ], "a list that was cut says how much of how much it is"


async def test_a_search_over_a_file_too_big_to_hold_is_refused_with_its_size(story_with: Scripted) -> None:
    """A search holds one file's text in memory, so what this build will not keep it will not scan (§0.6, R10)."""
    story = await story_with(calling("search", {"path": "big.log", "pattern": "needle"}), text("shame"))
    _ = _file(story, "big.log", "x" * (MAX_TEXT_CHARS + 1_000))

    await story.require("document")

    await story.type("find the needle")

    told_ = await _told(story)
    assert told_.startswith(f"big.log is {MAX_TEXT_CHARS + 1_000} characters, more than this build holds"), told_
    assert "read it in windows" in told_, "and the answer says what to do instead"


async def test_a_search_shows_the_lines_around_a_hit(story_with: Scripted) -> None:
    story = await story_with(
        calling("search", {"path": "notes.txt", "pattern": "two", "context_lines": 1}), text("thanks")
    )
    _ = _file(story, "notes.txt", "one\ntwo\nthree\nfour\n")

    await story.require("document")

    await story.type("and a line either side")

    assert (await _told(story)).splitlines() == [
        "1 match in notes.txt",
        "1- one",
        "2: two",
        "3- three",
    ], "a hit is marked with a colon and the company it keeps with a dash"


async def test_a_pattern_that_is_not_a_regex_is_the_arguments(story_with: Scripted) -> None:
    story = await story_with(calling("search", {"path": "notes.txt", "pattern": "("}), text("my mistake"))
    _ = _file(story, "notes.txt", "one\n")

    await story.require("document")

    await story.type("search with a broken pattern")

    told = await _told(story)
    assert told.startswith("invalid regex: "), told
    assert "missing" in told or "unbalanced" in told.lower(), "the regex engine's own words, not a rewrite"


async def test_a_search_of_a_kept_document_reads_what_was_kept(story_with: Scripted) -> None:
    content = _log()
    story = await story_with(
        calling("read", {"path": "big.log"}),
        _by_handle("search", {"pattern": "^line 7$"}),
        text("there it is"),
    )
    _ = _file(story, "big.log", content)

    await story.require("document")

    await story.type("read the log, then look inside what was kept")
    await story.finished()

    assert _told_of(story.requests()[-1]) == "1 match in d1\n8: line 7", (
        "the handle is searched where its content lives, and the hits are lines of it"
    )


def _log(lines: int = 4000) -> str:
    """Text well over what a result may send in one go, and readable enough to assert a window of."""
    return "\n".join(f"line {index}" for index in range(lines))


async def test_a_picture_is_kept_and_never_read_as_text(story_with: Scripted) -> None:
    story = await story_with(calling("read", {"path": "shot.png"}), text("I see"))
    picture = _picture(story, "shot.png")

    await story.type("look at the screenshot")

    told = await _told(story)
    assert told == (
        f"read shot.png (image/png, {len(picture)} bytes) — kept as d1\n"
        f"[d1] image/png, {len(picture)} bytes ({len(picture)} chars)"
    ), "no mojibake, no guessed text: the bytes are kept and the answer says what they are"
    assert "\ufffd" not in told
    assert any("document d1 kept" in line for line in story.screen()), story.screen()


async def test_a_picture_read_back_says_what_it_is_and_that_it_cannot_be_shown(story_with: Scripted) -> None:
    story = await story_with(
        calling("read", {"path": "shot.png"}),
        _by_handle("read", {}),
        text("ah"),
    )
    picture = _picture(story, "shot.png")

    await story.type("read the picture again by its handle")
    await story.finished()

    assert _told_of(story.requests()[-1]) == (
        f"d1 is image/png, {len(picture)} bytes; the content is kept whole, but a picture cannot be "
        "shown yet, so say what you need in words"
    ), "a window over a picture is not a window, so the answer that would have been a lie is not given"


def _picture(story: Story, name: str) -> bytes:
    """A file whose bytes say it is a PNG, put there before the story says anything."""
    data = b"\x89PNG\r\n\x1a\n" + bytes(range(24))
    _ = (story.workspace / name).write_bytes(data)
    return data


def _by_handle(tool: str, extra: dict[str, object]) -> Step:
    """A reply that calls *tool* on the handle the model was just given.

    The id exists only after the call that kept it, so it is read out of the text the model was handed —
    which is the only place a model would see it either.
    """

    def decide(request: Request) -> Reply:
        found = re.search(r"\[(d\d+)\]", told(request))
        assert found is not None, f"the result carried no handle to read: {told(request)!r}"
        return calling(tool, {"document_id": found.group(1), **extra})

    return decide
