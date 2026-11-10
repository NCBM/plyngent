"""What the model reads is projected in one place, so it can be asserted in one place."""

from __future__ import annotations

from plyngent.kernel.render import render, render_body, render_part, visible
from plyngent.kernel.toolresult import Document, Hint, ProcessControl, Text, ToolResult


def test_a_terminal_sequence_in_a_body_is_written_out_rather_than_executed() -> None:
    assert visible("a\x1b[2Jb") == "a\\x1b[2Jb", "ESC would clear somebody's screen on the way through"
    assert visible("bell\x07") == "bell\\x07", "a C0 control is a name, not a command"
    assert visible("del\x7f") == "del\\x7f"
    assert visible("tab\there\nand\rnewline") == "tab\there\nand\rnewline", "shaping stays shaping"
    assert visible("") == ""


def test_a_text_part_is_sent_as_it_is_apart_from_what_a_terminal_would_act_on() -> None:
    assert render_part(Text(value="plain")) == "plain"
    assert render_part(Text(value="\x1b[31mred")) == "\\x1b[31mred"


def test_a_document_part_is_a_handle_and_the_line_that_stands_in_for_it() -> None:
    part = Document(document_id="call-7", summary="40 KB of pytest output", chars=40_000)

    assert render_part(part) == "[call-7] 40 KB of pytest output (40000 chars)"


def test_a_command_is_laid_out_as_fields_and_then_its_streams() -> None:
    part = ProcessControl(argv=("pytest", "-q"), cwd="src", exit_code=1, stdout="", stderr="2 failed")

    assert render_part(part).splitlines() == [
        "exit_code=1",
        "timed_out=false",
        "cwd=src",
        "cmd=pytest -q",
        "--- stdout ---",
        "",
        "--- stderr ---",
        "2 failed",
    ]


def test_the_way_a_nested_command_got_here_is_part_of_what_the_model_reads() -> None:
    part = ProcessControl(argv=("true",), depth=2, via="then", fed=True, exit_code=0)

    lines = render_part(part).splitlines()

    assert lines[:4] == ["depth=2", "via=then", "exit_code=0", "timed_out=false"]
    assert "pipe=true" in lines


def test_a_merged_stream_is_one_block_because_it_was_one_stream() -> None:
    part = ProcessControl(argv=("sh", "-c", "echo o; echo e >&2"), errout="o\ne", mix_stderr=True)

    rendered = render_part(part)

    assert "--- output ---" in rendered, "there is one stream, so there is one heading"
    assert "--- stderr ---" not in rendered
    assert rendered.endswith("o\ne")


def test_a_hint_is_the_call_that_would_work_and_why_it_would() -> None:
    part = Hint(tool="write_file", given=(("overwrite", "true"),), why="replace it on purpose")

    assert render_part(part) == "hint: write_file(overwrite=true) — replace it on purpose"
    assert render_part(Hint(tool="read_file")) == "hint: read_file", "a step with nothing to patch is the call alone"
    assert render_part(Hint(tool="read_file", why="try the other one")) == "hint: read_file — try the other one"


def test_the_body_is_the_parts_in_the_order_they_happened() -> None:
    body = (Text(value="first"), Text(value="second"))

    assert render_body(body) == "first\nsecond"
    assert render_body(()) == ""


def test_the_model_reads_the_summary_and_then_the_body() -> None:
    assert render(ToolResult.ok("ran 2 commands", body=(Text(value="a"), Text(value="b")))) == ("ran 2 commands\na\nb")
    assert render(ToolResult.ok("nothing to say")) == "nothing to say", "no body, no dangling newline"
