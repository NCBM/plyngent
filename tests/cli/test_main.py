"""Which lane a run is on, decided from the streams in front of it.

``_editor`` is the one place the program looks at its own standard input, and ``_markdown`` the one place
it looks at its standard output, so they are the places a test can hold a terminal up to it. A fake
terminal is a class whose ``isatty`` says yes — that is the whole of what the choice reads, and the point
of the test is that nothing else decides it.
"""

from __future__ import annotations

import importlib.metadata
import io
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from plyngent import config
from plyngent.__main__ import _address, _announcements, _brought, _editor, _markdown, _tree_of, main
from plyngent.boot import endpoint_for
from plyngent.cli.announce import Desktop
from plyngent.cli.grammar import Catalog
from plyngent.cli.line import PlainEditor, PromptEditor
from plyngent.kernel.providing import ToolSet
from plyngent.wire.manifest import WorkspaceRow


class _Terminal:
    """A stream that answers ``isatty`` the way a terminal does, and answers nothing else."""

    def isatty(self) -> bool:
        return True


def test_a_pipe_is_read_plainly() -> None:
    chosen = _editor(session="main", catalog=Catalog(), stream=io.StringIO())

    assert isinstance(chosen, PlainEditor), "a prompt drawn on a pipe is either a guess or a hang"


def test_a_terminal_is_read_by_the_prompt() -> None:
    chosen = _editor(session="main", catalog=Catalog(), stream=_Terminal())

    assert isinstance(chosen, PromptEditor)


def test_a_run_can_ask_for_the_plain_lane_whatever_its_input_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYNGENT_PLAIN", "1")

    chosen = _editor(session="main", catalog=Catalog(), stream=_Terminal())

    assert isinstance(chosen, PlainEditor), "a supervisor may hold a terminal and still want no prompt"


def test_saying_no_to_the_plain_lane_leaves_the_choice_to_the_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLYNGENT_PLAIN", "0")

    chosen = _editor(session="main", catalog=Catalog(), stream=_Terminal())

    assert isinstance(chosen, PromptEditor), "unset and off are the same thing: the stream decides"


def test_an_answer_is_laid_out_as_markdown_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLYNGENT_PLAIN", raising=False)

    assert _markdown(reading=_Terminal(), written=_Terminal()) is True


def test_a_stream_that_is_not_a_terminal_gets_the_text_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PLYNGENT_PLAIN", raising=False)

    assert _markdown(reading=_Terminal(), written=io.StringIO()) is False, (
        "an answer wrapped for a width nobody asked for is a different answer, and a window is escape codes"
    )
    assert _markdown(reading=io.StringIO(), written=_Terminal()) is False


def test_the_plain_lane_reads_markdown_as_the_text_it_is(monkeypatch: pytest.MonkeyPatch) -> None:
    """One word decides both halves: the lane that was told to read plainly does not lay an answer out."""
    monkeypatch.setenv("PLYNGENT_PLAIN", "1")

    assert _markdown(reading=_Terminal(), written=_Terminal()) is False


def test_a_tree_a_run_names_is_the_one_it_means(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--workspace .`` is a choice: it resolves the way every path in this program does (E154)."""
    monkeypatch.chdir(tmp_path)

    assert _tree_of(Path(), said=(), at=Path("somewhere.sock")) == str(tmp_path.resolve()), (
        "a directory named — a bare `.` — is a tree meant, whether or not this kernel has opened it"
    )


def test_a_run_standing_in_a_served_tree_means_that_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Already being in a tree a kernel serves is saying which tree you mean — no question, no guess."""
    served = tmp_path / "served"
    inside = served / "inside"
    inside.mkdir(parents=True)
    monkeypatch.chdir(inside)

    assert _tree_of(None, said=(WorkspaceRow(path=str(served)),), at=Path("somewhere.sock")) == str(served)


def test_a_run_with_nobody_to_ask_refuses_rather_than_waits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipe has no keyboard: a run that must choose says which trees there were (E154)."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO())

    with pytest.raises(click.ClickException) as raised:
        _tree_of(None, said=(WorkspaceRow(path=str(elsewhere)),), at=Path("somewhere.sock"))

    assert str(elsewhere) in str(raised.value), "the answer names what this run could have meant"
    assert "--workspace" in str(raised.value), "and what it would have taken to say it"


def test_a_run_with_a_terminal_is_asked_which_tree_it_means(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Somebody at the keyboard is asked, and what they pick is what the connection means."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", _Terminal())
    offered: list[list[str]] = []

    def answer(*_args: object, **kwargs: object) -> str:
        chosen = kwargs["type"]
        offered.append(list(chosen.choices))  # type: ignore[attr-defined]
        return str(elsewhere)

    monkeypatch.setattr("plyngent.__main__.click.prompt", answer)

    assert _tree_of(None, said=(WorkspaceRow(path=str(elsewhere)),), at=Path("somewhere.sock")) == str(elsewhere)
    assert offered == [[str(tmp_path.resolve()), str(elsewhere), "nothing"]], (
        "this directory, the trees the kernel named, or nothing at all"
    )


def test_a_question_is_put_elsewhere_only_where_this_run_has_somewhere_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[notify]`` has three answers and the third is read here, where the surface is (E180).

    A key left out is "whatever this build can do": a desktop when the extra is installed, and nothing at
    all otherwise. A person who wrote ``false`` gets nothing even where the extra is there, and the numbers
    the rules were given are the file's own.
    """
    said = config.Notify(grace_seconds=2.0, min_interval_seconds=7.0)
    monkeypatch.setattr("plyngent.__main__.available", lambda: True)

    chosen = _announcements(said)

    assert chosen is not None
    assert (chosen.grace_seconds, chosen.min_interval_seconds) == (2.0, 7.0)
    assert isinstance(chosen.surface, Desktop)
    assert _announcements(config.Notify(enabled=False)) is None, "a person who said no is answered with nothing"

    monkeypatch.setattr("plyngent.__main__.available", lambda: False)

    assert _announcements(said) is None, "and a build with no extra has nowhere to put a question either"


def test_an_address_a_run_named_is_the_one_it_serves_on() -> None:
    assert _address(Path("somewhere.sock")) == Path("somewhere.sock")


def test_a_run_that_named_no_address_serves_this_person_own(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One kernel is one address, and it is nobody's tree: a daemon serves every tree a client names."""
    monkeypatch.chdir(tmp_path)

    named = _address("")

    assert named == endpoint_for(), "the default is this person's own address, not a guess at a tree"
    assert named.name.endswith(".sock"), "a socket, named so a person looking at the directory can tell"
    assert not named.exists(), "asking where a kernel would listen is not something that happens on disk"


def test_a_daemon_cannot_be_told_which_tree_to_work_in() -> None:
    """``--serve`` opens no tree, so ``--workspace`` beside it is a usage error, not a silent extra."""
    result = CliRunner().invoke(main, ["--serve", "--workspace", "."])

    assert result.exit_code == 2
    assert "--serve opens no tree" in result.output


def test_the_version_it_says_is_the_one_that_was_installed() -> None:
    """One number, and it is the distribution's own — read when it is asked for, before anything else.

    A constant in this module would be a second copy of a number that is written where the build reads it,
    and the copy a person is told is the one that has to be right. Being eager, the answer arrives before a
    file is read, which is what makes it usable on a build that cannot start.
    """
    result = CliRunner().invoke(main, ["--version"], prog_name="plyngent")

    assert result.exit_code == 0
    assert result.output == f"plyngent, version {importlib.metadata.version('plyngent')}\n"


def test_a_plugin_named_where_there_is_no_kernel_is_a_usage_error() -> None:
    """``--connect`` has no kernel of its own, so a plugin there is a mistake and not a silent nothing."""
    result = CliRunner().invoke(main, ["--connect", "somewhere.sock", "--plugin", "acme"])

    assert result.exit_code == 2
    assert "--connect brings no kernel of its own" in result.output


def test_a_plugin_that_is_not_installed_is_a_mistake_in_what_was_typed() -> None:
    """A name nobody installed is exit code 2, with the names this program does have in the sentence."""
    with pytest.raises(click.UsageError, match="no plugin named 'nowhere' is installed"):
        _ = _brought(("nowhere",), (), debug=False)


def test_a_plugin_file_brings_the_declarations_its_enable_answers(tmp_path: Path) -> None:
    """One file is one module, and ``enable`` is the name its declarations are answered under."""
    written = tmp_path / "mine.py"
    _ = written.write_text(
        "from composat import Declaration\n"
        "from plyngent.kernel.providing import ToolSet\n"
        "def enable():\n"
        "    return (Declaration(ToolSet, instance='acme', kwargs={'providers': ()}),)\n",
        encoding="utf-8",
    )

    brought = _brought((), (str(written),), debug=False)

    assert [one.instance for one in brought] == ["acme"]
    assert brought[0].component_class is ToolSet


def test_a_plugin_file_may_say_which_name_answers(tmp_path: Path) -> None:
    """``PATH:NAME`` is the escape from one convention: a module may already have a function by another."""
    written = tmp_path / "mine.py"
    _ = written.write_text("def bring():\n    return ()\n", encoding="utf-8")

    assert _brought((), (f"{written}:bring",), debug=False) == ()


def test_a_plugin_file_that_raises_is_a_broken_install_and_not_a_usage_error(tmp_path: Path) -> None:
    """A file that will not load is exit code 1: the path was right, and what is in it was not."""
    written = tmp_path / "mine.py"
    _ = written.write_text("raise RuntimeError('no')\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["--plugin-file", str(written), "-p", "hi"])

    assert result.exit_code == 1
    assert "raised while it was being loaded" in result.output


def test_a_plugin_file_that_is_not_there_is_a_mistake_in_what_was_typed(tmp_path: Path) -> None:
    with pytest.raises(click.UsageError, match="is not a file"):
        _ = _brought((), (str(tmp_path / "nothing.py"),), debug=False)


def test_a_program_that_will_not_start_says_which_component_and_why(tmp_path: Path) -> None:
    """A tool set that raises while it is built is a program that does not start, and the line names it.

    The name is the declaration's own key, because that is the thing a person can go and look at — and the
    trouble is named by its type, which is the whole of what the ordinary answer says about somebody else's
    code (E179).
    """
    written = tmp_path / "mine.py"
    _ = written.write_text(
        "from composat import Declaration\n"
        "from plyngent.kernel.providing import ToolSet\n"
        "class Breaking(ToolSet):\n"
        "    def __init__(self, providers=(), *, origin=''):\n"
        "        raise RuntimeError('no tools for you')\n"
        "def enable():\n"
        "    return (Declaration(Breaking, instance='bad'),)\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["--plugin-file", str(written), "-p", "hi"])

    assert result.exit_code == 1, "a component this program needs and cannot build is exit code 1"
    assert "a tool set this program was given could not be built" in result.output
    assert "Breaking:bad: RuntimeError: no tools for you" in result.output
    assert "Traceback" not in result.output, "one line about it is the ordinary answer"


def test_debug_says_where_a_failure_came_from(tmp_path: Path) -> None:
    """``--debug`` prints the traceback behind the sentence, because somebody else's line is the answer.

    The flag is the command line's own: there is no variable behind it, so "why did this start differently"
    has one place to be read from (E179).
    """
    written = tmp_path / "mine.py"
    _ = written.write_text(
        "from composat import Declaration\n"
        "from plyngent.kernel.providing import ToolSet\n"
        "class Breaking(ToolSet):\n"
        "    def __init__(self, providers=(), *, origin=''):\n"
        "        raise RuntimeError('no tools for you')\n"
        "def enable():\n"
        "    return (Declaration(Breaking, instance='bad'),)\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["--debug", "--plugin-file", str(written), "-p", "hi"])

    assert result.exit_code == 1
    assert "Traceback (most recent call last)" in result.output
    assert "RuntimeError: no tools for you" in result.output


def test_a_plugin_that_answers_something_else_is_a_broken_install(tmp_path: Path) -> None:
    """The shape is one shape — declarations — so anything else is a second contract for one word."""
    written = tmp_path / "mine.py"
    _ = written.write_text("def enable():\n    return 42\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["--plugin-file", str(written), "-p", "hi"])

    assert result.exit_code == 1
    assert "a plugin answers a tuple of declarations" in result.output
