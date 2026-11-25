"""Where the conversation is, and what it cost: the rows a frontend works out for itself.

These are the frontend's own rows, not the kernel's, which is why a story is the only place they
can be read from the outside: :func:`plyngent.cli.render.render` says nothing about them, and a unit
test of a row cannot say whether the loop put it on the right stream.

Everything here reads the notices. A row this client worked out is not something the kernel said, so it
does not land among the kernel's own words — which is also what keeps a program driving this one over a
pipe reading exactly what the kernel said and nothing else (§5 A, E130).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from plyngent.kernel.lm import Caps
from plyngent.kernel.lm.route import ModelEntry, Route
from plyngent.kernel.model import Reply
from plyngent.kernel.model import Usage as Cost
from plyngent.kernel.trail import Usage as Spent
from plyngent_creative.model import calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from tests.stories.driver import Story

    type Scripted = Callable[..., Awaitable[Story]]

#: The characters a decorated lane draws with, and the ones a plain lane must never produce.
DRAWN = ("│", "╰", "╭", "┌", "└")


def _route(name: str, *, window: int = 0) -> Route:
    """One endpoint a story can name, with nothing behind it: the scripted model answers, not this."""
    return Route(
        name=name,
        dialect="messages",
        base=f"http://{name}.invalid/v1",
        models=(ModelEntry(name="m", caps=Caps(max_output=64, context_window=window)),),
    )


async def test_a_turn_says_where_the_conversation_is(story_with: Scripted) -> None:
    """Once where it begins and once where it ends, and both say the route this session is on.

    The route is the half a person cannot see anywhere else: ``/set-route`` is a command they may have typed
    in a different session, hours ago, and which route is answering is not a fact the conversation says.
    The first of the three rows is the model this client chose when it attached (E170).
    """
    story = await story_with(text("hello"), routes=(_route("claude"),), route="claude")

    await story.type("hi")
    await story.finished()

    said = [line for line in story.notices() if line.startswith("main")]
    assert len(said) == 3, "the model chosen at attach, and then one where the turn began and one where it ended"
    assert all("route claude" in line for line in said)
    assert all("workspace0" in line for line in said), "and the directory the work happens in"


async def test_a_turn_says_what_it_cost(story_with: Scripted) -> None:
    story = await story_with(Reply(text="hello", usage=Cost(prompt=1200, completion=30)))

    await story.type("hi")
    await story.finished()

    tally = [line for line in story.notices() if line.startswith("turn 1 ")]
    assert tally, story.notices()
    assert "1 step" in tally[0], "one request to the model, and 'one step', not '1 steps'"
    assert "1200+30 tokens" in tally[0], "what was sent and what came back"
    rows = [row.payload for row in await story.host.trail.entries(story.session) if isinstance(row.payload, Spent)]
    assert [(row.prompt, row.completion, row.cached) for row in rows] == [(1200, 30, None)], (
        "the same three numbers, written down where a reader who was not here can find them"
    )
    assert rows[0].route == "", "the session never chose one, so the row names none"


async def test_the_context_row_says_a_share_when_the_route_stated_a_window(story_with: Scripted) -> None:
    """A percentage needs a denominator, and the denominator is a fact the model stated.

    E128 kept this row at an absolute number while ``Caps`` had no window to divide by; the number lives
    on the route announcement, so a route that stated one gets the share and a route that did not gets the
    count alone, and not a percentage of something this client made up.

    The window is a real one and not a small fiction: a route whose window cannot even hold this
    program's standing prompt — a persona, the playbook and the tool declarations — is a route whose
    every turn ends at the wall (E149).
    """
    story = await story_with(
        Reply(text="hello", usage=Cost(prompt=12500, completion=10)),
        routes=(_route("roomy", window=50_000),),
        route="roomy",
    )

    await story.type("hi")
    await story.finished()

    said = [line for line in story.notices() if "context" in line]
    assert said, story.notices()
    assert all("context 12500/50000 (25%)" in line for line in said), said
    assert not [line for line in said if "tokens" in line], "two numbers and a share, not three claims"


async def test_a_route_that_never_said_its_window_gets_the_count_alone(story_with: Scripted) -> None:
    story = await story_with(
        Reply(text="hello", usage=Cost(prompt=250, completion=10)), routes=(_route("quiet"),), route="quiet"
    )

    await story.type("hi")
    await story.finished()

    said = [line for line in story.notices() if "context" in line]
    assert said, story.notices()
    assert all("context 250 tokens" in line for line in said), "no window, no percentage"


async def test_a_call_says_how_long_it_took(story_with: Scripted) -> None:
    story = await story_with(calling("write_file", {"path": "notes.txt", "content": "hi"}), text("written"))

    await story.type("write a note")
    await story.answer("y")
    await story.finished()

    cost = [line for line in story.screen() if line.startswith("took ")]
    assert len(cost) == 1, "one row under one call, on the stream the call itself is on"
    assert re.fullmatch(r"took \d+ ms", cost[0]), "the kernel's own measurement, and it is a number"


async def test_a_plain_lane_draws_nothing(story_with: Scripted) -> None:
    """No bar, no corner: a stream that is not a terminal reads words and never a border."""
    story = await story_with(calling("write_file", {"path": "notes.txt", "content": "hi"}), text("written"))

    await story.type("write a note")
    await story.answer("y")
    await story.finished()

    written = "\n".join((*story.screen(), *story.notices()))
    assert not [character for character in DRAWN if character in written], written
