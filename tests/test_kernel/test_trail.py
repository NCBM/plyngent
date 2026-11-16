"""The projection written down: who said what, in the words a person reads an export in.

:func:`~plyngent.kernel.trail.transcript` is a pure function of the messages a model would be given, so
it is tested as one — what an export contains is exactly what a request would have carried, and the only
decision left is how to say who said it.
"""

from __future__ import annotations

from plyngent.kernel.model import Assistant, Result, User
from plyngent.kernel.trail import transcript


def test_a_conversation_is_written_as_who_said_what() -> None:
    held = transcript((User(text="the question"), Assistant(text="the answer")))

    assert held == "## person\n\nthe question\n\n## assistant\n\nthe answer"


def test_a_tool_result_is_named_by_the_tool_that_made_it() -> None:
    held = transcript((Result(call="c1", tool="read", status="ok", text="32 lines"),))

    assert held == "## tool: read (ok)\n\n32 lines", "an export has no turn to say which call this was"


def test_a_message_with_no_words_is_still_a_turn_of_somebody_s() -> None:
    """A call with nothing said before it is a message with an empty text, and dropping it would lose the
    fact that the model spoke at all."""
    assert transcript((Assistant(text=""),)) == "## assistant"


def test_an_empty_conversation_is_empty_text() -> None:
    assert transcript(()) == ""
