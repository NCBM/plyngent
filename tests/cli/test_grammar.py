"""What a typed line means, as pure functions: a split, a value, and what completion offers beside it.

One line goes in and an :data:`~plyngent.cli.grammar.Intent` comes out — no loop, no kernel, no
terminal — which is what makes this file possible at all and is also the reason the grammar is a
module of its own. The two things it can get wrong that nobody would notice by hand are the ones the
tests below are about: where an argument ends (a space is sometimes part of a value), and what the
person is shown while they are still typing it.
"""

from __future__ import annotations

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from plyngent.cli.grammar import Catalog, Reads, Runs, SlashCompleter, Travels, Wrong, build, decode, help_text
from plyngent.wire.command import Rollback, SetRoute
from plyngent.wire.manifest import KernelManifest, command_specs
from plyngent.wire.query import Grants, Trajectory


def _catalog() -> Catalog:
    """The vocabulary this client derives from the kernel's own manifest."""
    return build(KernelManifest(commands=command_specs()), session="main")


def test_a_quoted_argument_is_one_argument() -> None:
    intent = decode('/model --route "a b"', _catalog(), session="main")

    assert intent == Runs(command=SetRoute(session="main", route="a b")), "a value with a space in it is one value"


def test_an_unclosed_quote_is_a_sentence_rather_than_a_crash() -> None:
    intent = decode('/model --route "a b', _catalog(), session="main")

    assert isinstance(intent, Wrong), "the person is still typing; the line is answered, not thrown"
    assert "quotation" in intent.text


def test_an_empty_argument_is_still_an_argument() -> None:
    intent = decode('/model --route ""', _catalog(), session="main")

    assert intent == Runs(command=SetRoute(session="main", route="")), (
        "a value the person wrote as empty is empty, not a missing option"
    )


def test_a_switch_is_a_word_without_a_value() -> None:
    intent = decode("/rollback --force", _catalog(), session="main")

    assert intent == Runs(command=Rollback(session="main", force=True)), (
        "a switch says itself; nothing is typed after it"
    )


def test_completion_of_a_word_says_what_the_word_does() -> None:
    offered = list(SlashCompleter(_catalog()).get_completions(Document("/ne"), CompleteEvent()))

    assert [one.text for one in offered] == ["/new"]
    assert offered[0].display_meta_text == _catalog().offers["open"].summary, (
        "the line beside the name is why it exists"
    )


def test_completion_of_an_option_says_what_the_option_takes() -> None:
    offered = list(SlashCompleter(_catalog()).get_completions(Document("/new --ses"), CompleteEvent()))

    assert [one.text for one in offered] == ["--session"]
    assert offered[0].display_meta_text == "the conversation to act on"


def test_a_command_whose_argument_no_line_can_carry_is_withheld() -> None:
    """A tuple of objects has no spelling on a line, and offering it would build a value the kernel refuses."""
    catalog = _catalog()

    assert "offer" in catalog.withheld, "the shape is what makes it unspellable, not the command"
    assert catalog.word("offer") is None, "and it is not reachable through any alias either"
    assert isinstance(decode("/offer --tools x", catalog, session="main"), Wrong), (
        "a half-parsed argument is worse than no word at all"
    )


def test_a_read_word_asks_instead_of_asking_for_something() -> None:
    """A word that reads comes back as a query, and its one argument is which conversation it means."""
    assert decode("/grants", _catalog(), session="main") == Reads(query=Grants(session="main"))
    assert decode("/grants --session other", _catalog(), session="main") == Reads(query=Grants(session="other"))
    assert decode("/history", _catalog(), session="main") == Reads(query=Trajectory(session="main")), (
        "two reads, one shape: the word is the query's tag and the session is what it is about"
    )


def test_the_word_that_moves_where_this_client_stands_sends_nothing() -> None:
    """Where a frontend is reading is its own state, so this word is a value here and no command at all."""
    assert decode("/travel --entry e7", _catalog(), session="main") == Travels(entry="e7")
    assert isinstance(decode("/travel", _catalog(), session="main"), Wrong), "a place has to be named"


def test_a_read_word_takes_an_argument_it_cannot_and_nothing_else() -> None:
    """The shape is the query's own: a read that grew a second argument would be a command."""
    intent = decode("/grants --path somewhere", _catalog(), session="main")

    assert isinstance(intent, Wrong)
    assert "--session" in intent.text


def test_the_listing_names_the_words_that_read() -> None:
    """A word a person cannot find is a word that does not exist, and the listing is where they look."""
    listed = help_text(_catalog())

    assert "/grants" in listed
    assert "Which paths outside its workspace" in listed, "and it says what asking it will tell them"
