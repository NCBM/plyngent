"""The route and the model that posts to it, against real servers on the loopback.

A route is where one request goes and what becomes of it when it does not arrive: the dialect's path, the
credential in the header the route names, the delay between tries, the timeout, and the one decision —
try again or not — that comes from the kind of trouble. None of that is visible in a body, so it is not a
codec's test; and only half of it is visible on a screen, so it is not yet a story. What is left is a
model and a server, and that is what these are: `Posted` asked directly, and a real HTTP server counting
what arrived.

The clock appears in exactly one place (:func:`test_a_server_that_never_answers_is_given_up_on`) and it is
a route's own timeout, set to a hundredth of a second: a story about waiting should cost what the wait is
made of, not the length of a default.
"""

from __future__ import annotations

import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm import Caps, Route, Routes
from plyngent.kernel.lm.route import ModelEntry, Posted
from plyngent.kernel.model import ModelFailed, Request, User
from tests.loopback import Answer, answered


def _request(route: str = "") -> Request:
    """One request, naming the route a session's setting would name."""
    return Request(messages=(User(text="hi"),), route=route, model="m", cache_key="main:0:0")


def _caps(**given: object) -> Caps:
    return Caps(max_output=64, **given)  # type: ignore[arg-type]


def _answered(body: dict[str, object], *, say: str = "hello") -> Answer:
    """A chat-completions answer that said *say* and nothing else."""
    return answered(body, say=say)


def _route(base: str, **given: object) -> Route:
    """One route to a test server, with the ceiling and the dialect a chat-completions body needs."""
    return Route(
        name="local",
        dialect="chat-completions",
        base=base,
        models=(ModelEntry(name="m", caps=_caps()),),
        **given,  # type: ignore[arg-type]
    )


async def test_a_route_posts_the_dialects_body_to_the_dialects_path(serve) -> None:
    """One request: the dialect's endpoint, the conversation, and the credential where the route said.

    The path is the dialect's because a compatible service answers ``/chat/completions`` and an
    Anthropic one answers ``/messages``; the credential is the route's because no two gateways agree on
    which header it travels in.
    """
    site = serve()
    base = site.says(_answered({}))
    route = _route(base, secret="s3cr3t", headers=(("X-Traced", "yes"),))

    reply = await Posted(Routes([route], default="local")).respond(_request())

    assert site.asked == ["/chat/completions"]
    assert reply.text == "hello"
    assert site.bodies[0]["model"] == "m"
    assert site.bodies[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert site.headers[0]["authorization"] == "Bearer s3cr3t"
    assert site.headers[0]["x-traced"] == "yes"
    assert site.headers[0]["content-type"] == "application/json"


async def test_a_busy_server_is_asked_again_and_a_refusal_is_not(serve) -> None:
    """The outcome table decides, and it decides the same way a tool's retry does.

    ``429`` is ``BUSY`` — the world can fix that — so the delays a route carries are spent asking again;
    ``401`` is ``PERMISSION``, which no amount of asking lifts, so the second attempt is not made. The
    server counts the requests, because "it did not try again" is the assertion with no screen.
    """
    site = serve()
    base = site.says(Answer(status=429, body=b'{"error": {"message": "slow down"}}'), _answered({}))
    route = _route(base, attempts=(0.0, 0.0))

    reply = await Posted(Routes([route], default="local")).respond(_request())

    assert reply.text == "hello"
    assert len(site.asked) == 2

    refusing = serve()
    base = refusing.says(Answer(status=401, body=b'{"error": {"message": "no key"}}'), _answered({}))
    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes([_route(base, attempts=(0.0, 0.0))], default="local")).respond(_request())

    assert told.value.failure is Failure.PERMISSION
    assert "no key" in told.value.says
    assert len(refusing.asked) == 1


async def test_a_server_that_never_answers_is_given_up_on(serve) -> None:
    """A timeout is its own kind, and it costs the timeout and not a clock anybody else chose."""
    site = serve()
    base = site.says(Answer(body=b"{}", sleepy=1.0), _answered({}))
    route = _route(base, timeout=0.05)

    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes([route], default="local")).respond(_request())

    assert told.value.failure is Failure.TIMEOUT
    assert "0.05 seconds" in told.value.says
    assert len(site.asked) == 1


async def test_a_dialect_this_build_does_not_speak_is_a_route_that_cannot_ask(serve) -> None:
    """A route naming a dialect nobody implemented is the configuration's mistake, said plainly.

    ``ARGS`` and not ``MISSING``: the route exists and was found; what it says it answers in is a word
    this build has no row for, which is a fact about how the program was started.
    """
    route = Route(
        name="odd",
        dialect="carrier-pigeon",
        base="http://127.0.0.1:1",
        models=(ModelEntry(name="m", caps=_caps()),),
    )

    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes([route], default="odd")).respond(_request())

    assert told.value.failure is Failure.ARGS
    assert "carrier-pigeon" in told.value.says


async def test_a_request_with_no_route_anywhere_is_a_missing_route() -> None:
    """A kernel handed no routes refuses to ask, and invents no endpoint."""
    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes()).respond(_request())

    assert told.value.failure is Failure.MISSING


def test_which_route_answers_is_the_sessions_own_or_the_hosts_default() -> None:
    """Three questions, three answers: a name that exists, a session that never chose, and one that does not.

    A name this build no longer has is *not* an error here — a configuration file changed and a
    conversation must not become unusable — so it falls back to the default, and the strict lookup
    (:meth:`Routes.named`) is what answers a person who asked for a route by name.
    """
    one = Route(
        name="one",
        dialect="chat-completions",
        base="http://one/v1",
        models=(ModelEntry(name="m", caps=_caps()),),
    )
    two = Route(
        name="two",
        dialect="messages",
        base="http://two/v1",
        models=(ModelEntry(name="m", caps=_caps(pictures=True)),),
    )
    table = Routes([one, two], default="two")

    assert table.resolve("one") is one
    assert table.resolve("") is two
    assert table.resolve("retired") is two
    assert table.named("retired") is None
    asked = table.asking("one", "m")
    assert asked is not None and asked.route is one and asked.caps.pictures is False
    nobody = table.asking("", "m")
    assert nobody is not None and nobody.route is two and nobody.caps.pictures is True
    assert Routes().asking("", "m") is None


def test_a_conversation_asks_for_a_model_and_the_route_answers_with_its_caps() -> None:
    """E170: an id the route lists brings that model's numbers, and an id it does not is nothing to ask.

    ``None`` and not a quieter answer: a conversation that never chose a model, or chose one the file no
    longer lists, has no request to make — the route holds no caps of its own to fall back on.
    """
    route = Route(
        name="one",
        dialect="messages",
        base="http://x",
        models=(
            ModelEntry(name="small", caps=Caps(max_output=64)),
            ModelEntry(name="big", caps=Caps(max_output=4096)),
        ),
    )
    table = Routes([route], default="one")

    big = table.asking("one", "big")
    assert big is not None and (big.model, big.caps.max_output) == ("big", 4096)
    assert table.asking("one", "retired") is None, "an id the file does not list is nothing to ask"
    assert table.asking("one", "") is None, "and so is never having chosen one"
