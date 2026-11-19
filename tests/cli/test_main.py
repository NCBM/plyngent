"""Which lane a run is on, decided from the streams in front of it.

``_editor`` is the one place the program looks at its own standard input, and ``_markdown`` the one place
it looks at its standard output, so they are the places a test can hold a terminal up to it. A fake
terminal is a class whose ``isatty`` says yes — that is the whole of what the choice reads, and the point
of the test is that nothing else decides it.
"""

from __future__ import annotations

import io
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from plyngent.__main__ import _address, _editor, _markdown, _tree_of, main
from plyngent.boot import endpoint_for
from plyngent.cli.grammar import Catalog
from plyngent.cli.line import PlainEditor, PromptEditor
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
    """``--serve`` opens no tree, so ``--workspace`` beside it is a usage error rather than a silent extra."""
    result = CliRunner().invoke(main, ["--serve", "--workspace", "."])

    assert result.exit_code == 2
    assert "--serve opens no tree" in result.output
