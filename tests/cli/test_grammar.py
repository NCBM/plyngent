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

from plyngent.cli.grammar import (
    TRAVELLED,
    Catalog,
    Help,
    Reads,
    Runs,
    SlashCompleter,
    Travels,
    Wrong,
    build,
    decode,
    help_text,
    word_help,
)
from plyngent.wire.command import Goal, Rollback, SetRoute, Submit
from plyngent.wire.event import GoalMode
from plyngent.wire.manifest import KernelManifest, command_specs
from plyngent.wire.query import Grants, Skills, Trajectory


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


def test_a_word_whose_argument_is_one_of_two_words_takes_it_bare() -> None:
    """E146: ``/goal on`` is the value, and ``/goal`` alone is the question, not the empty mode.

    A word whose whole argument is a closed set has no option for it: naming the option would make a person
    type a word they can already see in the help, and the value they type is the same value the wire
    carries — ``GoalMode``, not the string click read off the line.
    """
    catalog = _catalog()

    assert decode("/goal on", catalog, session="main") == Runs(command=Goal(session="main", mode=GoalMode.ON))
    assert decode("/goal", catalog, session="main") == Runs(command=Goal(session="main")), (
        "no word at all is the question, which is a mode nobody stated"
    )
    assert isinstance(decode("/goal sideways", catalog, session="main"), Wrong), (
        "a word the command does not have is a complaint, not a mode"
    )


def test_completion_of_a_bare_value_says_which_words_it_takes() -> None:
    offered = list(SlashCompleter(_catalog()).get_completions(Document("/goal o"), CompleteEvent()))

    assert [one.text for one in offered] == ["off", "on"], "and only the values that start with what was typed"


def test_a_switch_is_a_word_without_a_value() -> None:
    intent = decode("/rollback --force", _catalog(), session="main")

    assert intent == Runs(command=Rollback(session="main", force=True)), (
        "a switch says itself; nothing is typed after it"
    )


def test_a_path_marked_with_at_is_completed_from_the_workspace(tmp_path) -> None:
    """An ``@`` word is completed from the tree, and the line is left as typed — nothing is checked.

    Completion is a way to spell a path the model will be told about; whether it exists is the model's
    question to ask with a tool, so a name that is not there survives the line unchanged.
    """
    (tmp_path / "src").mkdir()
    _ = (tmp_path / "src" / "main.py").write_text("x", encoding="utf-8")
    _ = (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    completer = SlashCompleter(_catalog(), tmp_path)

    offered = list(completer.get_completions(Document("look at @"), CompleteEvent()))
    deeper = list(completer.get_completions(Document("look at @src/ma"), CompleteEvent()))

    assert [one.text for one in offered] == ["@notes.txt", "@src/"], "a directory is offered with its slash"
    assert [one.text for one in deeper] == ["@src/main.py"]
    assert decode("look at @nowhere.txt", _catalog(), session="main") == Runs(
        command=Submit(session="main", text="look at @nowhere.txt")
    ), "the word reaches the model as it was typed"


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


def test_the_word_about_the_program_takes_no_conversation_to_be_about() -> None:
    """A skill table is what this program found at startup, so there is no session in the question."""
    assert decode("/skills", _catalog(), session="main") == Reads(query=Skills())
    assert isinstance(decode("/skills --session other", _catalog(), session="main"), Wrong), (
        "and naming a conversation is not a smaller answer, it is a different question"
    )


def test_the_listing_names_the_words_that_read() -> None:
    """A word a person cannot find is a word that does not exist, and the listing is where they look."""
    listed = help_text(_catalog())

    assert "/grants" in listed
    assert "Which paths outside its workspace" in listed, "and it says what asking it will tell them"


def test_a_word_describes_itself_and_the_listing_answers_nothing_else() -> None:
    """E117: one answer per question. ``/help`` lists; a word's own ``--help`` describes that word.

    The old shape answered both — ``/help <word>`` — and that was two texts that had to be kept in step
    for one answer a person wanted, while the derived one was already there and cannot go stale.
    """
    catalog = _catalog()

    assert decode("/help", catalog, session="main") == Help(text=help_text(catalog))
    assert decode("/help open", catalog, session="main") == Wrong(
        text="/help takes no argument; a word describes itself, as /open --help"
    ), "and a word after it is told where the answer actually is"
    assert decode("/help nope", catalog, session="main") == Wrong(
        text="/help takes no argument; a word describes itself, as /open --help"
    ), "a word nobody has is not the one the sentence points at"
    assert decode("/open --help", catalog, session="main") == Help(text=word_help(catalog, "open"))


def test_the_words_with_no_command_behind_them_describe_themselves_too() -> None:
    """A read word and the one this client answers itself have no schema, so their two lines are theirs."""
    catalog = _catalog()

    for word in ("grants", TRAVELLED):
        helped = decode(f"/{word} --help", catalog, session="main")
        assert isinstance(helped, Help) and helped.text.startswith(f"/{word} "), helped
