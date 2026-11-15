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

import json
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, override

import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm import Caps, Route, Routes
from plyngent.kernel.lm.route import Posted
from plyngent.kernel.model import ModelFailed, Request, User

if TYPE_CHECKING:
    from collections.abc import Iterator


def _request(route: str = "") -> Request:
    """One request, naming the route a session's setting would name."""
    return Request(messages=(User(text="hi"),), route=route, cache_key="main:0:0")


def _caps(**given: object) -> Caps:
    return Caps(max_output=64, **given)  # type: ignore[arg-type]


@dataclass
class Answer:
    """What a server says: a status and a body, and whether it says it at all."""

    status: int = 200
    body: bytes = b""
    sleepy: float = 0.0


@dataclass
class Site:
    """One server: what it answers, in order, and every request that arrived."""

    answers: list[Answer] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    bodies: list[dict[str, object]] = field(default_factory=list)
    headers: list[dict[str, str]] = field(default_factory=list)
    port: int = 0

    def says(self, *answers: Answer) -> str:
        """Answer these in order, and hand back the base URL a route would name."""
        self.answers.extend(answers)
        return f"http://127.0.0.1:{self.port}"


def _handler(site: Site) -> type[BaseHTTPRequestHandler]:
    """The request handler for one server, as ``ThreadingHTTPServer`` wants it."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            said = self.rfile.read(length)
            site.asked.append(self.path)
            site.bodies.append(json.loads(said.decode("utf-8")) if said else {})
            site.headers.append({name.lower(): value for name, value in self.headers.items()})
            answer = site.answers.pop(0) if site.answers else Answer(status=500, body=b"{}")
            if answer.sleepy:
                time.sleep(answer.sleepy)
            self.send_response(answer.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer.body)))
            self.end_headers()
            self.wfile.write(answer.body)

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Say nothing: a test's server is not somebody's terminal."""

    return Handler


@pytest.fixture
def serve() -> Iterator[object]:
    """A factory of servers on the loopback, each on a port the OS picks, all stopped at the end."""
    running: list[ThreadingHTTPServer] = []

    def start() -> Site:
        site = Site()
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(site))
        server.daemon_threads = True
        site.port = int(server.server_address[1])
        running.append(server)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        return site

    try:
        yield start
    finally:
        for server in running:
            server.shutdown()
            server.server_close()


def _answered(body: dict[str, object], *, say: str = "hello") -> Answer:
    """A chat-completions answer that said *say* and nothing else."""
    said = {"choices": [{"message": {"content": say}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3}}
    return Answer(body=json.dumps(said).encode(), status=200)


def _route(base: str, **given: object) -> Route:
    """One route to a test server, with the ceiling and the dialect a chat-completions body needs."""
    return Route(name="local", dialect="chat-completions", base=base, model="m", caps=_caps(), **given)  # type: ignore[arg-type]


async def test_a_route_posts_the_dialects_body_to_the_dialects_path(serve) -> None:
    """One request: the dialect's endpoint, the conversation, and the credential where the route said.

    The path is the dialect's because a compatible service answers ``/chat/completions`` and an
    Anthropic one answers ``/messages``; the credential is the route's because no two gateways agree on
    which header it travels in.
    """
    site = serve()
    base = site.says(_answered({}))
    route = _route(base, key="secret", headers=(("X-Traced", "yes"),))

    reply = await Posted(Routes([route], default="local")).respond(_request())

    assert site.asked == ["/chat/completions"]
    assert reply.text == "hello"
    assert site.bodies[0]["model"] == "m"
    assert site.bodies[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert site.headers[0]["authorization"] == "Bearer secret"
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
    route = Route(name="odd", dialect="carrier-pigeon", base="http://127.0.0.1:1", model="m", caps=_caps())

    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes([route], default="odd")).respond(_request())

    assert told.value.failure is Failure.ARGS
    assert "carrier-pigeon" in told.value.says


async def test_a_request_with_no_route_anywhere_is_a_missing_route() -> None:
    """A kernel handed no routes refuses to ask rather than inventing an endpoint."""
    with pytest.raises(ModelFailed) as told:
        _ = await Posted(Routes()).respond(_request())

    assert told.value.failure is Failure.MISSING


def test_which_route_answers_is_the_sessions_own_or_the_hosts_default() -> None:
    """Three questions, three answers: a name that exists, a session that never chose, and one that does not.

    A name this build no longer has is *not* an error here — a configuration file changed and a
    conversation must not become unusable — so it falls back to the default, and the strict lookup
    (:meth:`Routes.named`) is what answers a person who asked for a route by name.
    """
    one = Route(name="one", dialect="chat-completions", base="http://one/v1", model="m", caps=_caps())
    two = Route(name="two", dialect="messages", base="http://two/v1", model="m", caps=_caps(vision=True))
    table = Routes([one, two], default="two")

    assert table.resolve("one") is one
    assert table.resolve("") is two
    assert table.resolve("retired") is two
    assert table.named("retired") is None
    assert table.caps("one").vision is False
    assert table.caps("two").vision is True
    assert Routes().resolve("") is None
    assert Routes().caps("") == Caps(max_output=0)
