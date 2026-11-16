"""What a question answers when there is nobody to answer it, and why that is not one answer.

Two ways of nobody: a deadline arriving is an answer of a kind — the question falls back to what its
level said it would — and nobody being there at all is never a yes. Both are written down once, in
:func:`plyngent.wire.ask.unanswered`, and the two ends of it are what this file checks.
"""

from __future__ import annotations

import msgspec
import pytest

from plyngent.kernel.events import NoSink
from plyngent.wire.ask import (
    SILENCE,
    Agreed,
    AnyAnswer,
    By,
    Confirmation,
    FreeText,
    Picked,
    Secret,
    Selection,
    Severity,
    Written,
    unanswered,
)


@pytest.mark.parametrize("severity", list(Severity), ids=str)
async def test_nobody_is_never_a_yes_whatever_the_level(severity: Severity) -> None:
    """A question put to nobody was not allowed, and the light level does not change that."""
    agreed = await NoSink().confirm(prompt="run the tests?", severity=severity)

    assert agreed.accepted is False
    assert agreed.by is By.NOBODY, "the record says nobody was asked, not that somebody said no"


async def test_nobody_also_wrote_nothing_and_picked_nothing() -> None:
    sink = NoSink()

    assert await sink.text(prompt="what next?") == Written(by=By.NOBODY)
    assert await sink.secret(prompt="the password?") == Written(by=By.NOBODY), "and no default is suggested"
    assert await sink.choose(prompt="which?", options=("a", "b")) == Picked(by=By.NOBODY)


@pytest.mark.parametrize("severity", list(Severity), ids=str)
def test_a_deadline_falls_back_to_what_the_level_said(severity: Severity) -> None:
    """The connection fills ``timeout_default`` from the level, and this is the one place it is read."""
    question = Confirmation(id=7, severity=severity, timeout_default=SILENCE[severity])

    agreed = unanswered(question)

    assert isinstance(agreed, Agreed), "a yes-or-no question is answered with a yes or a no"
    assert agreed.id == 7, "the answer names the question it belongs to"
    assert agreed.accepted is SILENCE[severity], "the light question is a yes; the two heavier ones are not"
    assert agreed.by is By.TIMEOUT


def test_the_other_kinds_of_question_have_no_fallback_to_read() -> None:
    """A line and a choice have no "and if nobody speaks" — the absence of one is the answer."""
    assert unanswered(FreeText(id=3)) == Written(id=3, by=By.TIMEOUT)
    assert unanswered(Secret(id=4)) == Written(id=4, by=By.TIMEOUT)
    assert unanswered(Selection(id=5)) == Picked(id=5, by=By.TIMEOUT)


def test_an_answer_travels_with_who_gave_it() -> None:
    """The record says a person, a policy by name, a deadline, or nobody — and never says nothing."""
    politer = Agreed(id=3, accepted=True, by=By.POLICY, policy="away")

    assert msgspec.json.decode(msgspec.json.encode(politer), type=AnyAnswer) == politer
    with pytest.raises(msgspec.ValidationError):
        _ = msgspec.json.decode('{"answer": "agreed", "id": 3, "accepted": true}', type=AnyAnswer)
