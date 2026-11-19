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

from plyngent.__main__ import _address, _claim, _editor, _markdown
from plyngent.boot import endpoint_for
from plyngent.cli.grammar import Catalog
from plyngent.cli.line import PlainEditor, PromptEditor


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


def test_the_workspace_next_to_connect_is_a_claim_and_not_a_wish(tmp_path: Path) -> None:
    """E123: a kernel that works somewhere else is refused before this program sends anything."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert _claim(tmp_path, said=(str(tmp_path),)) is None, "the directory the kernel names is the one asked for"
    with pytest.raises(click.ClickException):
        _claim(tmp_path, said=(str(elsewhere),))


def test_a_claim_is_about_any_of_the_trees_a_kernel_names(tmp_path: Path) -> None:
    """A greeting lists the trees a kernel has open, so a directory it names is one it is working in."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    assert _claim(elsewhere, said=(str(tmp_path), str(elsewhere))) is None


def test_connect_without_a_workspace_claims_nothing(tmp_path: Path) -> None:
    assert _claim(None, said=(str(tmp_path),)) is None, "a run that named no directory reads what it finds"


def test_an_address_a_run_named_is_the_one_it_serves_on(tmp_path: Path) -> None:
    assert _address(Path("somewhere.sock"), workspace=tmp_path) == Path("somewhere.sock")


def test_a_run_that_named_no_address_serves_this_workspace_own(tmp_path: Path) -> None:
    """``--serve`` with no path is "this tree", and the name it uses is the one the tree already has."""
    named = _address("", workspace=tmp_path)

    assert named == endpoint_for(tmp_path), "the default is the workspace's own address, not a guess"
    assert named.name.endswith(".sock"), "a socket, named so a person looking at the directory can tell"
    assert not named.exists(), "asking where a kernel would listen is not something that happens on disk"


def test_two_workspaces_with_one_name_are_still_two_addresses(tmp_path: Path) -> None:
    mine = tmp_path / "one" / "project"
    yours = tmp_path / "two" / "project"
    mine.mkdir(parents=True)
    yours.mkdir(parents=True)

    assert _address("", workspace=mine) != _address("", workspace=yours), "a shared directory name is not an address"
