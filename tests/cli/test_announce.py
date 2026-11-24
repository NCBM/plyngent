"""The questions this client says out loud, and what a press on one answers (E180).

Two facts per test: whether a question is worth announcing, and — when it is — what it looks like away
from the terminal and what pressing one of its buttons means. The surface here is a stub that says
nothing: these are the rules, and they do not depend on a desktop.
"""

from __future__ import annotations

import math

from plyngent.cli.announce import NO, YES, Announcements, Choice, Wording
from plyngent.wire.ask import Agreed, By, Confirmation, FreeText, Picked, Secret, Selection, Written


class _Silent:
    """A surface that is never used: the rules decide, and nothing is said anywhere."""

    async def announce(self, told: Wording) -> str | None:
        raise AssertionError("the rules do not announce anything")

    def withdraw(self) -> None:
        raise AssertionError("nothing was announced, so nothing is taken back")


def _rules(*, grace: float = 10.0, interval: float = 30.0) -> Announcements:
    """The rules a client with a desktop was given, with the two numbers a test can move."""
    return Announcements(surface=_Silent(), grace_seconds=grace, min_interval_seconds=interval)


def test_a_secret_is_never_announced() -> None:
    """A value must not travel through a surface that outlives the question that asked for it."""
    question = Secret(prompt="the api key?", timeout=600.0)

    assert _rules().speaks(question, since=math.inf) is False
    assert _rules().words(question) is None
    assert _rules().answered(question, "anything") is None


def test_a_question_that_expires_inside_the_grace_is_not_announced() -> None:
    """A routine question waits five seconds, so it never reaches a desktop: the notice would arrive late."""
    rules = _rules(grace=10.0)

    assert rules.speaks(Confirmation(prompt="?", timeout=5.0), since=math.inf) is False
    assert rules.speaks(Confirmation(prompt="?", timeout=10.0), since=math.inf) is False, (
        "a deadline the grace would reach is the same as one it passes"
    )
    assert rules.speaks(Confirmation(prompt="?", timeout=10.5), since=math.inf) is True
    assert rules.speaks(Confirmation(prompt="?"), since=math.inf) is True, "no deadline of its own is no reason to wait"


def test_a_second_question_inside_the_interval_is_suppressed_and_not_kept_for_later() -> None:
    """A burst is answered by the one the person is already looking at (E180)."""
    rules = _rules(interval=30.0)
    question = Confirmation(prompt="?", timeout=600.0)

    assert rules.speaks(question, since=29.9) is False
    assert rules.speaks(question, since=30.0) is True
    assert rules.speaks(question, since=math.inf) is True


def test_a_yes_or_no_question_carries_two_buttons_and_a_press_on_one_is_the_person_s_yes() -> None:
    rules = _rules()
    question = Confirmation(id=4, prompt="write src/x.py?", timeout=600.0)

    told = rules.words(question)

    assert told == Wording(
        body="write src/x.py?", choices=(Choice(label="yes", value=YES), Choice(label="no", value=NO))
    )
    assert rules.answered(question, YES) == Agreed(id=4, accepted=True, by=By.HUMAN), (
        "where it was pressed is this client's own fact, and the answer is still the person's"
    )
    assert rules.answered(question, NO) == Agreed(id=4, accepted=False, by=By.HUMAN)
    assert rules.answered(question, "maybe") is None, "a word that is not one of the buttons answers nothing"


def test_a_question_about_code_names_the_head_of_it_and_how_much_is_left() -> None:
    """A notice is not a reader: it says what is being asked about, and stops (E191)."""
    rules = _rules()
    code = "".join(f"line {one}\n" for one in range(10))
    question = Confirmation(prompt="run this?", executable_code=code, timeout=600.0)

    told = rules.words(question)

    assert told is not None
    assert told.body.splitlines()[:2] == ["run this?", "line 0"]
    assert told.body.endswith("… 2 more lines"), "the count is what sends a person to the terminal it waits at"
    short = Confirmation(prompt="run this?", executable_code="print(1)\n", timeout=600.0)
    assert rules.words(short) == Wording(
        body="run this?\nprint(1)", choices=(Choice(label="yes", value=YES), Choice(label="no", value=NO))
    ), "a whole program that fits is shown whole, with nothing to count"


def test_a_small_choice_carries_one_button_per_option() -> None:
    rules = _rules()
    question = Selection(id=7, prompt="keep going?", options=("again", "stop"), timeout=600.0)

    told = rules.words(question)

    assert told is not None
    assert [one.label for one in told.choices] == ["again", "stop"], "one press is one option, and it says which"
    assert told.reply is False
    assert rules.answered(question, "stop") == Picked(id=7, value="stop", by=By.HUMAN)
    assert rules.answered(question, "later") is None, "an option nobody offered is not one to press"


def test_a_choice_with_too_many_options_or_several_answers_is_named_and_not_offered() -> None:
    """One press is one option, so a list of presses has no button that means it (E180)."""
    rules = _rules()
    many = Selection(prompt="which one?", options=("a", "b", "c", "d"), timeout=600.0)
    several = Selection(prompt="which ones?", options=("a", "b"), multiple=True, timeout=600.0)

    for question in (many, several):
        told = rules.words(question)
        assert told is not None
        assert told.choices == (), "nothing to press: the person answers this one at the terminal"
        assert told.body.startswith(question.prompt) and "a" in told.body.splitlines()[1]


def test_prose_gets_a_reply_box_and_what_is_typed_is_the_answer() -> None:
    rules = _rules()
    question = FreeText(id=2, prompt="what next?", timeout=600.0)

    told = rules.words(question)

    assert told == Wording(body="what next?", reply=True)
    assert rules.answered(question, " run the tests ") == Written(id=2, text="run the tests", by=By.HUMAN)
    assert rules.answered(question, "") == Written(id=2, text="", by=By.HUMAN), (
        "an empty line is a legitimate answer to prose, the same as at the prompt"
    )
