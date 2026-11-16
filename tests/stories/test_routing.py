"""Routing: what one asking costs, what a person is told when it could not be made, and which route
answers.

A story reads the consequences of a route without a route: the model is still scripted, and what is
scripted is the *answer* a route would have produced — including the answer "I could not ask". That keeps
these stories about the turn's end and the screen, which is what a story is for, while the HTTP side of a
route is tested where a body and a status can be asserted (``tests/test_kernel/test_route.py``).

The other two faces a route has are here because they are exactly the two a story can see: **what the
model was asked** (which route, and which cache key) and **what the screen said** about it. The cost of a
request and the fact that an answer was cut off are here for the same reason — neither has a place in the
history.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm import Caps
from plyngent.kernel.lm.route import Route
from plyngent.kernel.model import Picture, Reply, Result, Usage
from plyngent.kernel.trail import Attempt
from tests.stories.driver import Story
from tests.stories.model import Failing, ScriptedModel, calling, text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    type Scripted = Callable[..., Awaitable[Story]]


def _route(name: str, dialect: str = "messages", *, vision: bool = False) -> Route:
    """One endpoint a story can name, with nothing behind it: the scripted model answers, not this."""
    return Route(
        name=name,
        dialect=dialect,
        base=f"http://{name}.invalid/v1",
        model="m",
        caps=Caps(max_output=64, vision=vision),
    )


#: A picture, as the bytes a file would hold: the signature is what says what it is.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24


async def test_a_turn_that_could_not_ask_says_which_kind_of_trouble_it_met(story_with: Scripted) -> None:
    """A model that could not be asked ends the turn with the kind and the sentence, and the session goes on.

    The kind is the word a program switches on and the sentence is what a person reads, and both have to
    arrive: a terminal that only said "the turn failed" is where ``main`` ended up, with the reason
    somewhere in a traceback nobody was allowed to print.
    """
    story = await story_with(
        Failing(Failure.BUSY, "local answered 429: slow down"),
        text("and then it worked"),
    )

    await story.type("a first thought")
    await story.type("a second thought")

    assert story.screen()[:3] == (
        "turn 1 started (chat)",
        "turn 1 failed (busy): local answered 429: slow down",
        "turn 2 started (chat)",
    ), "the failure is reported and the session is not held by it"
    assert story.screen()[-1] == "turn 2 finished"
    assert len(story.requests()) == 2, "the second turn really asked again"


async def test_what_one_asking_cost_is_on_the_screen(story_with: Scripted) -> None:
    """The numbers §10.5 exists for, said out loud: what was sent, what came back, and what was cached.

    ``cached`` is shown only when the endpoint reported one, because a provider that said nothing and a
    provider that reported a miss are different facts and one line cannot carry both.
    """
    story = await story_with(
        Reply(text="costly", usage=Usage(prompt=1200, completion=30, cached=900)),
        Reply(text="silent", usage=Usage(prompt=1300, completion=20)),
    )

    await story.type("first")
    await story.type("second")

    assert "turn 1 tokens 1200+30 (900 cached)" in story.screen()
    assert "turn 2 tokens 1300+20" in story.screen()
    assert "turn 2 tokens 1300+20 (" not in "".join(story.screen()), "a silent provider says nothing"


async def test_an_answer_cut_at_the_limit_says_so(story_with: Scripted) -> None:
    """An answer that hit its output limit is one a person is told about, before the turn's own end.

    The cut is not the end of the turn — the history keeps what was said and the next round continues from
    it — so the line arrives between the words and the finish, which is where a reader needs it.
    """
    story = await story_with(Reply(text="half a sen", cut=Failure.LIMIT))

    await story.type("say something long")

    assert story.screen()[:3] == (
        "turn 1 started (chat)",
        "turn 1 answer was cut short (limit)",
        "half a sen",
    ), "the cut is said before the text, because that is when it is worth knowing"


async def test_a_session_is_asked_by_the_route_it_names(story_with: Scripted) -> None:
    """The name a session chose is what reaches the model, and changing it is one line on the screen.

    The route is read at the boundary of each request, which is what makes a change legal mid-turn and
    what makes the assertion possible at all: the story reads the request the model was asked, and the
    route is a field of it.
    """
    story = await story_with(text("first"), text("second"), routes=[_route("one"), _route("two")], route="one")

    await story.type("a thought")
    await story.type("/model --route two")
    await story.type("another thought")

    assert story.requests()[0].route == "one", "a session that never chose asks the host's default"
    assert story.requests()[1].route == "two"
    assert "route: two (epoch 1)" in story.screen(), "the change is said out loud, with its epoch"


async def test_a_route_that_does_not_exist_is_refused_and_nothing_moves(story_with: Scripted) -> None:
    """A name this kernel does not have is answered where it was asked, and the session keeps asking.

    No third state: a route either exists here or is refused, and a refusal writes nothing — the next
    request goes where the last one went.
    """
    story = await story_with(text("first"), text("second"), routes=[_route("one")], route="one")

    await story.type("a thought")
    await story.type("/model --route nowhere")
    await story.type("another thought")

    assert "route refused: no route named 'nowhere' here" in story.screen()
    assert [asked.route for asked in story.requests()] == ["one", "one"]


async def test_the_prefix_key_names_the_session_and_moves_with_its_prompt(story_with: Scripted) -> None:
    """Three requests, three keys: what changed is the two counters, and the session is in the key.

    §10.5's whole accounting rests on this number: a provider hashes the prefix, we name it, and a miss
    is attributable to the tool block or to the plan because they are two numbers and not one.
    """
    story = await story_with(text("one"), text("two"), text("three"), routes=[_route("one")], route="one")

    await story.type("a thought")
    await story.type("/compact --text settled")
    await story.require("fs")
    await story.type("another thought")

    assert [asked.cache_key for asked in story.requests()] == ["main:1:0", "main:2:1"], (
        "the client's own tools moved the set once when it attached; the summary and the"
        " disclosure moved each counter once more"
    )


async def test_the_route_a_session_chose_outlives_a_restart(tmp_path: Path) -> None:
    """A route is a setting, not a command's effect: reopening the conversation reopens the choice.

    Without this a restarted client would quietly fall back to the default model, and the conversation
    would continue in a voice nobody asked for.
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"
    routes = [_route("one"), _route("two")]

    first = await Story.start(root, model=ScriptedModel([text("first")]), routes=routes, route="one", journal=journal)
    try:
        await first.type("a thought")
        await first.type("/model --route two")
    finally:
        await first.close()

    second = await Story.start(root, model=ScriptedModel([text("second")]), routes=routes, route="one", journal=journal)
    try:
        await second.type("another thought")

        assert second.requests()[0].route == "two", "the session remembered which route it chose"
        assert second.requests()[0].cache_key == f"main:{second.toolset_epoch}:0", (
            "the counter resumes from the journal too: a reopened session claims the prefix it knows"
        )
    finally:
        await second.close()


async def _attempts(story: Story) -> tuple[Attempt, ...]:
    """Every attempt row this session's journal holds, found the way a reader would: as its roots.

    An attempt is written with no parent, so it sits beside the conversation instead of in it — which is
    why the roots are where to look, and why finding one here also proves it cannot have become a place
    the conversation continues from.
    """
    trail = story.host.trail
    shape = await trail.shape(story.session)
    rows = [await trail.entry(story.session, one) for one in shape.roots]
    return tuple(row.payload for row in rows if row is not None and isinstance(row.payload, Attempt))


async def test_a_failed_attempt_is_a_row_a_restart_can_read(tmp_path: Path) -> None:
    """A request nobody answered is written down, with the route epoch that was in force.

    R4 asked for attempts to be first-class, and this is their first writer: the setting says which route
    a session asks *now*, and the row says which one it asked when it tried — the only way "which model
    was that" is answerable after a restart.
    """
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root,
        model=ScriptedModel([Failing(Failure.BUSY, "one answered 429: slow down")]),
        routes=[_route("one"), _route("two")],
        route="one",
        journal=journal,
    )
    try:
        await first.type("a thought")
        await first.type("/model --route two")
    finally:
        await first.close()

    second = await Story.start(
        root,
        model=ScriptedModel([text("later")]),
        routes=[_route("one"), _route("two")],
        route="one",
        journal=journal,
    )
    try:
        (attempt,) = await _attempts(second)

        assert attempt.turn == 1
        assert attempt.route == "one", "the route it asked, not the one the session chose afterwards"
        assert attempt.epoch == 0, "and the epoch that route stood at"
        assert (attempt.failure, attempt.retry) == ("busy", "same_call"), (
            "the closed-set words, so a row outlives a change to either set"
        )
        assert attempt.says == "one answered 429: slow down"
    finally:
        await second.close()


async def test_an_attempt_beside_the_chain_does_not_move_where_the_conversation_continues(story_with: Scripted) -> None:
    """The row is a root, so a conversation that ended in a failure continues exactly where it was.

    The failure arrives after the person's line was written, so the history is one entry — and the
    attempt, written beside it, must not become the tip a restarted client would continue from.
    """
    story = await story_with(Failing(Failure.TIMEOUT, "local had not answered"), text("after"))

    await story.type("a thought")
    await story.type("another thought")

    assert story.requests()[1].messages[0].text == "a thought", "the second turn continued the chain"
    assert len(story.requests()[1].messages) == 2, "one line and one answer, and no row in between"


async def _read_a_picture(story: Story) -> Result:
    """Put a picture in the workspace, read it, and answer the message the model was left with.

    Waits for the turn to end rather than for the loop to have got somewhere: reading a picture is a
    file read, which a tool hands to a thread, and the request the answer belongs to is assembled after
    that thread reports back.
    """
    (story.workspace / "pic.png").write_bytes(PNG)
    await story.type("look at the picture")
    await story.finished()
    last = story.requests()[-1].messages[-1]
    assert isinstance(last, Result), "the second request ends with what the read came back with"
    return last


async def test_a_route_that_cannot_see_is_told_about_a_picture_in_words(story_with: Scripted) -> None:
    """A blind route reads the handle and not the pixels, and nothing is fetched on its behalf.

    The words are true either way — the format, the size, and where it went — which is why `read` no
    longer says a picture cannot be shown: down one route it can, and down another the sentence would be
    a lie the same result had to live with.
    """
    story = await story_with(
        calling("read", {"path": "pic.png"}),
        text("say it in words"),
        routes=[_route("blind")],
        route="blind",
    )

    seen = await _read_a_picture(story)

    assert seen.pictures == (), "a model that cannot read pixels is not sent any"
    assert seen.documents == ("d1",), "the handle is still the machine-readable half of the answer"
    assert "kept as d1" in seen.text


async def test_a_route_that_can_see_is_given_the_bytes(story_with: Scripted) -> None:
    """A route that can see is handed the image itself, fetched from the store at request time (E56).

    Both gates have to be open: the route's model must read pictures *and* its dialect must have a channel
    for one. The third case in this story is the second gate on its own — a seeing model behind a
    text-only wire — and it is answered the same way as a blind one.
    """
    story = await story_with(
        calling("read", {"path": "pic.png"}),
        text("what is it?"),
        routes=[_route("seeing", vision=True)],
        route="seeing",
    )

    seen = await _read_a_picture(story)

    assert seen.pictures == (Picture(media="image/png", data=PNG),), "the bytes, with the type they are"
    assert seen.documents == ("d1",), "and the handle stays, because the row is what a restart reads"

    text_only = await story_with(
        calling("read", {"path": "pic.png"}),
        text("what is it?"),
        routes=[_route("chatty", dialect="chat-completions", vision=True)],
        route="chatty",
    )
    unwired = await _read_a_picture(text_only)
    assert unwired.pictures == (), "a dialect with no image channel does not get a picture either"
