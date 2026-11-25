"""A model that thinks before it answers: what a person reads, and what the next round is told.

Thinking is two things at once, and the second is the one that is not a preference: a round that thought
and then asked for a tool has to carry its thinking into the next request, or a thinking endpoint refuses
the request. So the story that matters asserts the *second* request, and the ones about the screen assert
that a thought is readable as what it is, not mixed into the answer (E145, §0.13 E94).

What a thought is not: a message. It is a field of the answer the model gave, so it belongs to that
round, not between the speakers — which is why a pipe gets the answer and nothing else, and why
the two screenings below are two lanes, not one screen read twice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Assistant, Call, Reply, Thought
from plyngent_creative.model import Streamed, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]


async def test_a_pipe_is_given_the_answer_and_not_the_thinking(story_with: Scripted) -> None:
    """A block of thinking with nothing to mark it off would read as something the model said.

    So the lane that cannot say *which* of the two a line is writes only the one a script came for. The
    thought is still in the journal and still on its way back to the model; what it is not is part of the
    text a pipe is reading.
    """
    story = await story_with(Reply(text="the answer", thoughts=(Thought(text="looking first"),)))

    await story.type("say something")
    await story.finished()

    assert story.screen() == (
        "turn 1 started (chat)",
        "the answer",
        "turn 1 finished",
    ), "a lane with one colour has no way to say which half of the answer a line came from"


async def test_a_screen_is_shown_what_the_model_thought_before_its_answer(story_with: Scripted) -> None:
    """The other lane: a thought is a block of its own, and it comes before the answer it led to."""
    story = await story_with(Reply(text="the answer", thoughts=(Thought(text="looking first"),)), markdown=True)

    await story.type("say something")
    await story.finished()

    read = "\n".join(story.screen())
    assert "looking first" in read, "the thinking is drawn, and in its own kind of line"
    assert read.index("looking first") < read.index("the answer"), "before the answer, because it came first"


async def test_a_thought_is_read_as_lines_while_it_is_still_being_thought(story_with: Scripted) -> None:
    """The same screen, arriving a piece at a time: a line appears when it is whole, as an answer's does."""
    story = await story_with(
        Streamed(
            thought=("one ", "two\nthree"),
            pieces=("four",),
            reply=Reply(text="four", thoughts=(Thought(text="one two\nthree"),)),
        ),
        markdown=True,
    )

    await story.type("say something")
    await story.finished()

    read = "\n".join(story.screen())
    assert "one two" in read and "three" in read, "the thought is gathered into lines the way an answer is"
    assert read.index("three") < read.index("four"), "and it is finished before what it led to begins"
    assert "one two\nthree" not in read, "a line is not written twice, once per fragment"


async def test_a_round_that_thought_hands_its_thinking_to_the_next_one(story_with: Scripted) -> None:
    """The acceptance: the second request carries the first round's thinking, signature and all.

    A model that thinks and then asks for a tool is the case a thinking endpoint refuses when the
    thinking went missing, so this is asserted on the request.
    """
    story = await story_with(
        Reply(
            text="checking",
            calls=(Call(id="c1", tool="glob_paths", arguments='{"pattern":"*"}'),),
            thoughts=(Thought(text="the directory first", signature="s1"),),
        ),
        text("done"),
    )

    await story.type("what is here?")
    await story.finished()

    first, second = story.requests()
    assert second.messages[1] == Assistant(
        text="checking",
        calls=(Call(id="c1", tool="glob_paths", arguments='{"pattern":"*"}'),),
        thoughts=(Thought(text="the directory first", signature="s1"),),
    ), "the round is given back as it was, because the endpoint reads it that way"
    assert len(first.messages) == 1, "and the request before it had one message and no thinking to give back"
