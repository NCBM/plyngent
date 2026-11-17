"""What a session owes a model between two requests — the object, and the rules it keeps.

This is not a thing a story can show. A story sees the line the model was finally given, and the whole
point of this object is the gap before that: which sentences survive a second fire, in which order they
are said, and which of them is not said at all because it says the same thing again. Those are questions
about a value with nobody watching, so they are asserted here; that a turn writes the line at a boundary
is a story's, and it is told where a writer exists to fire one.
"""

from __future__ import annotations

import pytest

from plyngent.kernel.notices import Notices, Pending
from plyngent.kernel.trail import NOTICE_MARKER


def test_a_line_is_written_with_the_marker_and_nothing_else_is_added_to_it() -> None:
    assert Pending(text="2 things are left").body == f"{NOTICE_MARKER}\n2 things are left"
    assert Pending(text="").body == "", "a reminder that stopped being true says nothing to a model"


def test_the_lines_are_said_in_the_order_the_writers_were_declared() -> None:
    """Not in the order they fired: what a model reads has to be a function of the state, not of timing.

    Two writers whose fire order is the reverse of their declaration is the only case that tells the two
    apart, so it is the case asserted.
    """
    notices = Notices(("plan", "limit"))
    notices.say("limit", "the last answer was cut short")
    notices.say("plan", "2 things are left")

    assert [(key, pending.text) for key, pending in notices.owed()] == [
        ("plan", "2 things are left"),
        ("limit", "the last answer was cut short"),
    ]


def test_a_writer_that_fires_twice_owes_one_sentence_and_it_is_the_last_one() -> None:
    notices = Notices(("plan",))
    notices.say("plan", "3 things are left")
    notices.say("plan", "2 things are left")

    assert [(key, pending.text) for key, pending in notices.owed()] == [("plan", "2 things are left")]


def test_a_sentence_already_said_is_not_owed_again() -> None:
    notices = Notices(("plan",))
    notices.say("plan", "2 things are left")
    _ = notices.owed()
    notices.say("plan", "2 things are left")

    assert notices.owed() == (), "a reminder that still holds is not news to a model that read it"


def test_a_reminder_that_stopped_holding_is_owed_once_and_as_nothing() -> None:
    notices = Notices(("plan",))
    notices.say("plan", "2 things are left")
    _ = notices.owed()
    notices.say("plan", "")
    retold = notices.owed()

    assert [(key, pending.body) for key, pending in retold] == [("plan", "")], "the row is there and says nothing"
    notices.say("plan", "")
    assert notices.owed() == (), "and saying it twice is not saying it again"


def test_a_key_nobody_declared_is_refused() -> None:
    """A line under a key no flush consults would be a sentence nobody ever says — an error, not a row."""
    notices = Notices(("plan",))

    with pytest.raises(ValueError, match="no notice writer named 'plan2'"):
        notices.say("plan2", "something")
