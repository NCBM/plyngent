"""The questions this client answers for a person, and the lines it leaves for having done it (§0.15).

The kernel asks and is answered; which of the two answered is the frontend's business, and these are
the frontend's own rules — a level it may speak at, a kind of question it may never speak for, and the
one line each answer leaves on the screen.
"""

from __future__ import annotations

from plyngent.cli.policy import AWAY, YOLO, Answered, Policy, Tally
from plyngent.wire.ask import Agreed, By, Confirmation, FreeText, Picked, Secret, Selection, Severity


def test_a_policy_answers_yes_or_no_questions_and_nothing_else() -> None:
    """A permission is what both names were promises about: words and choices are still the person's."""
    everything = Policy(yolo=True, away=True, away_danger=True)

    assert everything.answers(Secret(prompt="the api key?")) is None, "a password is never somebody's policy"
    assert everything.answers(FreeText(prompt="what next?")) is None
    assert everything.answers(Selection(prompt="which one?", options=("a", "b"))) is None
    assert everything.answers(Confirmation(prompt="write it?")) is not None


def test_yolo_says_yes_to_the_two_light_levels_and_stops() -> None:
    yolo = Policy(yolo=True)

    assert yolo.answers(Confirmation(prompt="?", severity=Severity.ROUTINE)) == Answered(accepted=True, policy=YOLO)
    assert yolo.answers(Confirmation(prompt="?", severity=Severity.CONFIRM)) == Answered(accepted=True, policy=YOLO)
    assert yolo.answers(Confirmation(prompt="?", severity=Severity.DANGER)) is None, (
        "it is not a promise to run programs nobody was asked about"
    )


def test_away_answers_each_level_as_its_own_silence_would_have() -> None:
    """The light question is a yes while it waits; the ordinary one is not (§0.15, E112 and E115)."""
    away = Policy(away=True)

    assert away.answers(Confirmation(prompt="?", severity=Severity.ROUTINE)) == Answered(accepted=True, policy=AWAY)
    assert away.answers(Confirmation(prompt="?", severity=Severity.CONFIRM)) == Answered(accepted=False, policy=AWAY)
    assert away.answers(Confirmation(prompt="?", severity=Severity.DANGER)) == Answered(accepted=False, policy=AWAY)


def test_answering_a_dangerous_question_with_a_yes_is_a_second_switch() -> None:
    """ "I am not here" and "I am not here and you may run anything" are different things to have said."""
    answered = Policy(away=True, away_danger=True).answers(Confirmation(prompt="?", severity=Severity.DANGER))

    assert answered == Answered(accepted=True, policy=AWAY)


def test_both_on_at_once_is_the_first_one_with_an_answer() -> None:
    """``--yolo`` is consulted first because it is the one that says yes; away never overwrites it."""
    both = Policy(yolo=True, away=True)

    assert both.answers(Confirmation(prompt="?", severity=Severity.CONFIRM)) == Answered(accepted=True, policy=YOLO), (
        "away would have refused this one, and it does not get the chance to"
    )
    assert both.answers(Confirmation(prompt="?", severity=Severity.DANGER)) == Answered(accepted=False, policy=AWAY), (
        "the heaviest question is away's, because yolo has nothing to say about it"
    )


def test_an_answer_carries_the_policy_that_gave_it() -> None:
    question = Confirmation(id=9, prompt="?", severity=Severity.ROUTINE)

    answered = Policy(yolo=True).answers(question)
    assert answered is not None

    reply = answered.reply(question)
    assert reply == Agreed(id=9, accepted=True, by=By.POLICY, policy=YOLO), (
        "a yes from a program is not a yes from a person, and the record says which"
    )


def test_a_choice_that_names_its_own_default_is_answered_with_it() -> None:
    """E150: the question says what its silence picks, so a policy repeats it rather than choosing.

    This is the one narrowing of "an option nobody offered is not one a policy may pick": the option is
    in front of the person already — it is what an empty line gives — so answering it is not a policy
    deciding on their behalf. Both switches answer it, and ``--yolo`` first, for the reason it is first
    everywhere else.
    """
    question = Selection(prompt="keep going?", options=("again", "stop"), default="again")

    answered = Policy(away=True).answers(question)
    assert answered == Answered(accepted=True, policy=AWAY, picked="again")
    assert answered is not None and answered.reply(question) == Picked(id=0, value="again", by=By.POLICY, policy=AWAY)

    assert Policy(yolo=True).answers(question) == Answered(accepted=True, policy=YOLO, picked="again")
    assert Policy(yolo=True, away=True).answers(question) == Answered(accepted=True, policy=YOLO, picked="again"), (
        "the first one with an answer gives it, which is the rule a confirmation already follows"
    )


def test_a_choice_with_nothing_in_front_of_the_person_stays_theirs() -> None:
    """No default, or a default that is not one of the options: nobody is offered anything, so nobody speaks."""
    everything = Policy(yolo=True, away=True, away_danger=True)

    assert everything.answers(Selection(prompt="?", options=("a", "b"))) is None
    assert everything.answers(Selection(prompt="?", options=("a", "b"), default="c")) is None, (
        "a suggested line that is not an option is not an offer either"
    )


def test_the_stretch_ends_with_a_count_and_nothing_at_all_when_nothing_was_answered() -> None:
    tally = Tally()

    assert tally.said() == "", "a run where nobody answered says nothing about policies"
    tally.keep(Answered(accepted=True, policy=YOLO))
    tally.keep(Answered(accepted=True, policy=AWAY))
    tally.keep(Answered(accepted=False, policy=AWAY))

    assert tally.said() == "yolo: 1 answered (1 allowed, 0 refused); away: 2 answered (1 allowed, 1 refused)"
