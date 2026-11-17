"""The net group: one request that leaves from this machine, and what a person is asked about it.

Every story here is against a **real** server on the loopback. The tool is an HTTP client, and a story
that stubbed one out would be a story about the stub: what is being tested is that a status, a header, a
redirect and a stream of bytes come back through the same three layers as everything else.

Each server keeps the record of what it was asked for, because half of what an agreement means is the
request that was *not* made — "nothing arrived at the server" is an assertion no screen can carry.

The model is scripted, so the call is the story's: the arguments are what a model would have written and
what comes back is what it would have read.
"""

from __future__ import annotations

import ipaddress
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, override

import pytest

from plyngent.kernel.model import Result
from plyngent.kernel.net.web import Engine, EngineFailed, Engines, Hit, Page
from plyngent.kernel.toolresult import Status
from plyngent.wire.ask import Grades, Severity
from tests.stories.driver import Story
from tests.stories.model import ScriptedModel, calling, text, told

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

    from plyngent.kernel.model import Request
    from plyngent.kernel.net.client import Gate

    type Scripted = Callable[..., Awaitable[Story]]

#: A picture, as the bytes a response would carry: the signature is what says what it is.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24

#: Something that is not text and is not a picture: the case a model must never be handed as mojibake.
BLOB = b"\x00\x01\x02\x03"

TEXT_TYPE = "text/plain; charset=utf-8"


@dataclass
class Answer:
    """What a server says for one path."""

    status: int = 200
    body: bytes = b""
    content_type: str = TEXT_TYPE
    headers: tuple[tuple[str, str], ...] = ()


@dataclass
class Site:
    """One server: what it answers, on which port, and what it was asked for."""

    routes: dict[str, Answer] = field(default_factory=dict)
    asked: list[tuple[str, str]] = field(default_factory=list)
    posts: list[bytes] = field(default_factory=list)
    port: int = 0

    def serve(
        self,
        path: str,
        body: str | bytes = "",
        *,
        status: int = 200,
        content_type: str = TEXT_TYPE,
        headers: tuple[tuple[str, str], ...] = (),
    ) -> str:
        """Answer *path* with this, and hand back the URL a call would name."""
        said = body.encode("utf-8") if isinstance(body, str) else body
        self.routes[path] = Answer(status=status, body=said, content_type=content_type, headers=headers)
        return f"http://127.0.0.1:{self.port}{path}"

    def pointed(self, path: str, *, to: str, status: int = 302) -> str:
        """Answer *path* by sending the request to *to*."""
        return self.serve(path, status=status, headers=(("Location", to),))


def _handler(site: Site) -> type[BaseHTTPRequestHandler]:
    """The request handler for one server, as the class ``ThreadingHTTPServer`` wants."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def answered(self) -> None:
            """Answer with what this site says about the path, recording the request as it came in."""
            path = self.path.split("?", 1)[0]
            site.asked.append((self.command, path))
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                site.posts.append(self.rfile.read(length))
            said = site.routes.get(path, Answer(status=404, body=b"no such page"))
            self.send_response(said.status)
            self.send_header("Content-Type", said.content_type)
            self.send_header("Content-Length", str(len(said.body)))
            for name, value in said.headers:
                self.send_header(name, value)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(said.body)

        def do_GET(self) -> None:
            self.answered()

        def do_POST(self) -> None:
            self.answered()

        def do_PUT(self) -> None:
            self.answered()

        def do_DELETE(self) -> None:
            self.answered()

        def do_HEAD(self) -> None:
            self.answered()

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Say nothing: a story's server is not somebody's terminal."""

    return Handler


@pytest.fixture
def serve() -> Iterator[Callable[[], Site]]:
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


def _result(request: Request) -> Result:
    """The result one round ended with, which is the only place the status is written down."""
    said = request.messages[-1]
    assert isinstance(said, Result), "the answer after a call carries what came of it"
    return said


async def _run(story: Story, *, agree: int = 0) -> None:
    """Show the net group, send the story's message, agree to *agree* questions, and wait for the end."""
    await story.require("net")
    await story.type("go and read the page")
    for _ in range(agree):
        await story.answer("y")
    await story.finished()


def _call(url: str, **given: object) -> dict[str, object]:
    """The arguments a model would write for one call: the URL, why, and whatever else the story wants."""
    return {"url": url, "intention": "read the page", **given}


async def test_a_private_host_is_asked_about_once_and_then_remembered(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/index.html", "hello from the library")
    story = await story_with(calling("client_fetch", _call(page)), calling("client_fetch", _call(page)), text("done"))

    await _run(story, agree=2)

    assert _reasons(story, 0) == [
        f"127.0.0.1:{site.port} is a loopback/private address (127.0.0.1)",
        "http: anybody on the way can read it",
    ], "one question, and every reason in it"
    assert _reasons(story, 1) == ["http: anybody on the way can read it"], (
        "the second call names the same machine at the same port, so that reason is gone: it was agreed to"
    )
    assert story.host.net.grants == {f"127.0.0.1:{site.port}"}, "what was agreed to is the host and the port"
    assert told(story.requests()[1]) == (f"200 GET {page} — 22 bytes text\nhello from the library"), (
        "the line says everything but the body, and the body is the body"
    )
    assert site.asked == [("GET", "/index.html"), ("GET", "/index.html")], "and both calls went out"


async def test_a_declined_read_is_a_refusal_and_sends_nothing(story_with: Scripted, serve: Scripted) -> None:
    """Acceptance of §0.15's E118: a person saying no is the same decision for a read and a write.

    Nothing was unavailable, so trying the same call again would ask the same person the same question.
    """
    site = serve()
    page = site.serve("/index.html", "hello")
    story = await story_with(calling("client_fetch", _call(page)), text("fair"))

    await story.require("net")
    await story.type("go and read the page")
    await story.decline()
    await story.finished()

    assert told(story.requests()[-1]) == (
        f"nobody agreed to contact 127.0.0.1:{site.port}, so the request was not sent"
    ), "a request nobody agreed to is not made, and the model is told why"
    assert _result(story.requests()[-1]).status is Status.REFUSED, "a refusal is not a bad minute for the world"
    assert site.asked == [], "and nothing reached the server"


async def test_a_read_nobody_answered_is_worth_trying_again(story_with: Scripted, serve: Scripted) -> None:
    """Silence is not a decision: an unanswered read is the world being unavailable (§0.15, E118)."""
    site = serve()
    page = site.serve("/index.html", "hello")
    story = await story_with(calling("client_fetch", _call(page)), text("fair"), grades=Grades(danger=0.01))

    await story.require("net")
    await story.type("go and read the page")
    await story.finished()

    (question,) = story.questions()
    assert question.severity is Severity.DANGER, "an unanswered question is the one the level governs"
    assert _result(story.requests()[-1]).status is Status.FAILED, "nothing was refused: nobody spoke"
    assert site.asked == [], "and nothing reached the server"


async def test_a_write_nobody_answered_is_a_refusal_not_a_bad_minute(story_with: Scripted, serve: Scripted) -> None:
    """The other half of the pair: a write nobody agreed to is not replayed (§0.15, E118)."""
    site = serve()
    _ = site.serve("/echo", "said it back")
    story = await story_with(
        calling("client_fetch", _call(f"https://127.0.0.1:{site.port}/echo", method="POST")),
        text("fair"),
        grades=Grades(confirm=0.01),
    )
    story.host.net.assume_public = (ipaddress.ip_network("127.0.0.0/8"),)

    await story.require("net")
    await story.type("send it")
    await story.finished()

    assert _result(story.requests()[-1]).status is Status.REFUSED, (
        "asking the same question again is what a retry would do, and nobody is going to answer it"
    )
    assert site.asked == [], "a write nobody agreed to is not sent"


async def test_the_metadata_address_is_never_asked_about(story_with: Scripted) -> None:
    story = await story_with(
        calling("client_fetch", _call("http://169.254.169.254/latest/meta-data")),
        text("fair"),
    )

    await _run(story)

    assert story.questions() == (), "no grant lifts it, so there is nothing to ask"
    assert told(story.requests()[-1]) == (
        "169.254.169.254 resolves to 169.254.169.254, which this build never contacts"
    ), "the answer names the host and the address, and the request is never made"


async def test_a_scheme_that_is_not_http_is_an_argument_error(story_with: Scripted) -> None:
    story = await story_with(calling("client_fetch", _call("file:///etc/passwd")), text("fair"))

    await _run(story)

    assert told(story.requests()[-1]) == "url scheme must be one of http, https; got 'file'", (
        "the closed set is said, because the same mistake is fixable"
    )
    assert story.questions() == ()


async def test_every_hop_is_a_destination_of_its_own(story_with: Scripted, serve: Scripted) -> None:
    first, second = serve(), serve()
    landing = second.serve("/land", "the second page")
    start = first.pointed("/jump", to=landing)
    story = await story_with(calling("client_fetch", _call(start)), text("done"))

    await _run(story, agree=2)

    assert [question.prompt.split("?")[0] for question in story.questions()] == [
        f"may I send GET to {start}",
        f"may I send GET to {landing}",
    ], "a redirect is a second machine at a second port, so it is a second question"
    assert {question.severity for question in story.questions()} == {Severity.DANGER}, (
        "reaching the machine's own network is the level nothing here can undo"
    )
    assert story.host.net.grants == {f"127.0.0.1:{first.port}", f"127.0.0.1:{second.port}"}
    assert first.asked == [("GET", "/jump")], "the first server was asked once"
    assert second.asked == [("GET", "/land")], "and the one it pointed at, once"
    assert told(story.requests()[-1]) == (
        f"200 GET {landing} — 15 bytes text (cleartext HTTP at {landing}, 1 redirect)\nthe second page"
    )


async def test_a_followed_chain_says_how_it_got_there(story_with: Scripted, serve: Scripted) -> None:
    first, second = serve(), serve()
    landing = second.serve("/land", "the second page")
    start = first.pointed("/jump", to=landing)
    story = await story_with(calling("client_fetch", _call(start)), text("done"))

    await _run(story, agree=2)

    assert told(story.requests()[-1]).splitlines()[0] == (
        f"200 GET {landing} — 15 bytes text (cleartext HTTP at {landing}, 1 redirect)"
    ), "the summary says the chain was followed and that it was never protected on the way"


async def test_the_redirect_budget_is_a_limit_with_the_argument_that_lifts_it(
    story_with: Scripted, serve: Scripted
) -> None:
    site = serve()
    start = site.serve("/jump", status=302, headers=(("Location", "/land"),))
    _ = site.serve("/land", "the second page")
    story = await story_with(calling("client_fetch", _call(start, max_redirects=0)), text("fair"))

    await _run(story, agree=1)

    assert told(story.requests()[-1]) == (
        f"more than 0 redirects from {start}\nhint: client_fetch(max_redirects=10) — "
        "the chain is longer than this call allowed"
    ), "a budget that is reached is a limit, and the next value is offered as data"
    assert site.asked == [("GET", "/jump")], "and the hop it would not take was never made"


async def test_a_body_too_long_to_send_is_kept_and_continued_from(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/big.txt", "the log line\n" * 4000)
    story = await story_with(
        calling("client_fetch", _call(page)),
        lambda request: calling("read", {"document_id": _handle(told(request)), "limit": 1}),
        text("there it is"),
    )

    await _run(story, agree=1)

    assert told(story.requests()[1]).splitlines()[0] == (f"200 GET {page} — 52.0 KB text (kept)"), (
        "the summary says what happened and that the body is somewhere else"
    )
    assert told(story.requests()[-1]) == "read d1 (lines 1-1 of 4000)\nthe log line\n", (
        "and the handle resumes it with no second request"
    )


async def test_a_kept_body_is_still_there_after_a_restart(tmp_path, transport: str, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/big.txt", "the log line\n" * 4000)
    journal = tmp_path / "journal.db"
    root = tmp_path / "workspace"

    first = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("client_fetch", _call(page)), text("kept")]),
        journal=journal,
    )
    try:
        await first.require("net")
        await first.type("go and read the page")
        await first.answer("y")
        await first.finished()
        assert "[d1]" in told(first.requests()[-1]), "a body that did not fit is a handle"
        assert await first.stop() == 0
    finally:
        await first.close()

    second = await Story.start(
        root,
        transport=transport,
        model=ScriptedModel([calling("read", {"document_id": "d1", "limit": 1}), text("there it is")]),
        journal=journal,
    )
    try:
        await second.type("the first line of what I fetched")

        assert told(second.requests()[-1]) == "read d1 (lines 1-1 of 4000)\nthe log line\n", (
            "a handle names content, and the content is still there after a restart"
        )
    finally:
        await second.close()


async def test_a_picture_is_kept_whole_and_said_to_be_one(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    picture = site.serve("/cat.png", PNG, content_type="image/png")
    story = await story_with(
        calling("client_fetch", _call(picture)),
        lambda request: calling("read", {"document_id": _handle(told(request))}),
        text("fair"),
    )

    await _run(story, agree=1)

    assert told(story.requests()[1]).splitlines()[0] == (f"200 GET {picture} — 32 bytes image (kept)"), (
        "an image is bytes nobody can read as text, so it is kept and said to be one"
    )
    assert told(story.requests()[-1]).startswith("d1 is image/png, 32 bytes"), (
        "and this build can only say what it is, until a picture part exists"
    )


async def test_a_body_that_is_not_text_is_not_handed_over_as_mojibake(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    blob = site.serve("/blob", BLOB, content_type="application/octet-stream")
    story = await story_with(
        calling("client_fetch", _call(blob)),
        lambda request: calling("read", {"document_id": _handle(told(request))}),
        text("fair"),
    )

    await _run(story, agree=1)

    assert told(story.requests()[1]).splitlines()[0] == (f"200 GET {blob} — 4 bytes binary (kept)"), (
        "no NUL byte ever reaches the model as text"
    )
    assert told(story.requests()[-1]).startswith("d1 is application/octet-stream, 4 bytes"), (
        "the store keeps the type the response declared"
    )


async def test_the_download_cap_cuts_the_body_and_says_so(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/long.txt", "x" * 500)
    story = await story_with(calling("client_fetch", _call(page, max_bytes=100)), text("done"))

    await _run(story, agree=1)

    assert told(story.requests()[-1]) == (f"200 GET {page} — 100 bytes text (truncated)\n" + "x" * 100), (
        "a cut body is not a failure: the status is the answer and the cut is said beside it"
    )


async def test_a_method_that_changes_something_carries_its_body_and_is_asked_about(
    story_with: Scripted, serve: Scripted
) -> None:
    site = serve()
    echo = site.serve("/echo", "said it back")
    story = await story_with(calling("client_fetch", _call(echo, method="POST", body="hello")), text("done"))

    await _run(story, agree=1)

    assert _reasons(story) == [
        f"127.0.0.1:{site.port} is a loopback/private address (127.0.0.1)",
        "http: anybody on the way can read it",
        "POST changes something on the other end",
        "it carries a body, which leaves this machine",
    ], "the verb and the body are two reasons of the same kind, and both are named"
    assert site.posts == [b"hello"], "the body went out as it was written"
    assert told(story.requests()[-1]).endswith("\nsaid it back")


async def test_a_declined_write_is_not_replayed(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    echo = site.serve("/echo", "said it back")
    story = await story_with(calling("client_fetch", _call(echo, method="POST", body="hello")), text("fair"))

    await story.require("net")
    await story.type("go and read the page")
    await story.decline()
    await story.finished()

    assert told(story.requests()[-1]) == (
        f"nobody agreed to contact 127.0.0.1:{site.port}, so the request was not sent"
    ), "the same words a declined read gets"
    assert _result(story.requests()[-1]).status is Status.REFUSED, (
        "but a write nobody agreed to is a refusal, not a busy world: it is never replayed"
    )
    assert site.posts == [] and site.asked == []


async def test_a_head_request_has_no_body_to_read(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/index.html", "hello from the library")
    story = await story_with(calling("client_fetch", _call(page, method="HEAD", max_bytes=1)), text("done"))

    await _run(story, agree=1)

    assert site.asked == [("HEAD", "/index.html")], "the server was asked with the verb the call named"
    assert told(story.requests()[-1]) == f"200 HEAD {page} — 0 bytes empty", (
        "a HEAD has no body: the size says nothing arrived and the cap has nothing to cut"
    )


async def test_a_public_https_read_is_asked_about_by_nobody(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    _ = site.serve("/index.html", "hello")
    story = await story_with(calling("client_fetch", _call(f"https://127.0.0.1:{site.port}/index.html")), text("done"))
    # What a resolver's fake-IP pool does for Clash, a story does with one range: the loopback is public.
    story.host.net.assume_public = (ipaddress.ip_network("127.0.0.0/8"),)

    await _run(story)

    assert story.questions() == (), "a public https read has nobody to ask, which is the whole exemption"
    assert "could not be reached" in told(story.requests()[-1]), (
        "and the request is still made: this server speaks no TLS, so it fails as a protocol error"
    )


async def test_a_public_https_write_is_still_asked_about(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    _ = site.serve("/echo", "said it back")
    story = await story_with(
        calling("client_fetch", _call(f"https://127.0.0.1:{site.port}/echo", method="POST")), text("fair")
    )
    story.host.net.assume_public = (ipaddress.ip_network("127.0.0.0/8"),)

    await story.require("net")
    await story.type("go and read the page")
    await story.decline()
    await story.finished()

    assert [question.prompt.split("? ")[-1] for question in story.questions()] == [
        "POST changes something on the other end"
    ], "everything about this request would have been exempt but the verb"
    assert story.questions()[0].severity is Severity.CONFIRM, (
        "a body is a fact about this request and it can be sent again, so it is the ordinary level"
    )
    assert _result(story.requests()[-1]).status is Status.REFUSED
    assert site.asked == [], "a declined write is not sent"


async def test_cleartext_is_asked_about_every_time(story_with: Scripted, serve: Scripted) -> None:
    site = serve()
    page = site.serve("/index.html", "hello")
    story = await story_with(calling("client_fetch", _call(page)), calling("client_fetch", _call(page)), text("done"))
    story.host.net.assume_public = (ipaddress.ip_network("127.0.0.0/8"),)

    await _run(story, agree=2)

    assert len(story.questions()) == 2, (
        "plain http is a fact about this request and not about that machine, so it is never remembered"
    )
    assert {question.severity for question in story.questions()} == {Severity.DANGER}, (
        "what leaves in cleartext has already gone by the time anybody could regret it"
    )
    assert story.host.net.grants == set(), "nothing was granted: the only reason was the transport"
    assert site.asked == [("GET", "/index.html"), ("GET", "/index.html")]


@dataclass
class Source:
    """An engine a story scripts: the pages it answers with, or the reason it cannot answer.

    The verbs are what the tool asks of an engine, so a story's engine is a real implementation of them —
    the two engines this build ships (`deepseek`, `bing`) are exercised by their own unit tests instead,
    because reaching either of them needs a network this suite is not allowed to use.
    """

    hits: list[tuple[str, str, str]] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    refuses: str = ""

    @property
    def name(self) -> str:
        """What a failure calls this engine: the story's own word for what it is."""
        return type(self).__name__

    async def search(self, query: str, *, gate: Gate, on: str = "") -> list[Hit]:
        """Answer what the story put here, recording the query as it came in."""
        del gate, on  # a story's engine has no destination and no account to choose between
        self.asked.append(query)
        if self.refuses:
            raise EngineFailed(self.refuses)
        return [Hit(title=title, url=url, snippet=snippet) for title, url, snippet in self.hits]


class Broken(Source):
    """An engine that cannot answer: its name is what a person reads in the summary."""


class Answering(Source):
    """An engine that answers, or answers with nothing."""


@dataclass
class Reader:
    """A page reader a story scripts: the pages it can read, and the URLs it was asked for."""

    pages: dict[str, Page] = field(default_factory=dict)
    asked: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """What a failure calls this engine: the story's own word for what it is."""
        return type(self).__name__

    async def page(self, url: str, *, on: str = "") -> Page:
        """Answer with the page the story put under *url*; a URL nobody put there is the story's bug."""
        del on  # one account, so there is nothing to choose between
        self.asked.append(url)
        return self.pages[url]


def asking(*sources: Source) -> Engines:
    """The table a story hands the kernel: each scripted source's search verb, under its own name."""
    return Engines([Engine(name=source.name, search=source.search) for source in sources])


def paging(*readers: Reader) -> Engines:
    """The same, for the engines that read pages."""
    return Engines([Engine(name=reader.name, page=reader.page) for reader in readers])


async def test_a_search_shows_the_pages_an_engine_named(story_with: Scripted) -> None:
    source = Answering(hits=[("Python", "https://python.org", "the language"), ("Docs", "https://docs.python.org", "")])
    story = await story_with(calling("web_search", {"query": "python"}), text("done"), engines=asking(source))

    await story.require("net")
    await story.type("search for the language")
    await story.finished()

    assert told(story.requests()[-1]) == (
        "2 hits for 'python'\n- Python — https://python.org\n  the language\n- Docs — https://docs.python.org"
    ), "a hit is a place to read plus the one line the engine had for it"
    assert source.asked == ["python"], "and the query went out as it was written"
    assert story.questions() == (), "an engine that reaches no destination of its own asks nothing"


async def test_a_search_falls_through_and_says_what_each_engine_said(story_with: Scripted) -> None:
    broken = Broken(refuses="the index is down")
    empty = Answering()
    working = Answering(hits=[("Only", "https://example.com/only", "")])
    story = await story_with(
        calling("web_search", {"query": "python"}),
        text("done"),
        engines=asking(broken, empty, working),
    )

    await story.require("net")
    await story.type("search for the language")
    await story.finished()

    assert told(story.requests()[-1]) == (
        "1 hit for 'python' (Broken: the index is down; Answering: no results)\n- Only — https://example.com/only"
    ), "a broken engine does not take the search down, and what it said is carried out to the summary"
    assert [source.asked for source in (broken, empty, working)] == [["python"], ["python"], ["python"]], (
        "every engine before the one that answered was asked"
    )


async def test_a_search_that_never_answers_names_every_engine(story_with: Scripted) -> None:
    first, second = Broken(refuses="no key"), Broken(refuses="rate limited")
    story = await story_with(calling("web_search", {"query": "python"}), text("fair"), engines=asking(first, second))

    await story.require("net")
    await story.type("search for the language")
    await story.finished()

    assert told(story.requests()[-1]) == ("no hit for 'python' (Broken: no key; Broken: rate limited)"), (
        "a search with no answer is an answer, and it says who could not help"
    )


async def test_with_no_engine_the_web_tools_are_not_in_the_table(story_with: Scripted) -> None:
    story = await story_with(
        calling("tool_require", {"group": "net"}),
        calling("web_search", {"query": "python"}),
        calling("web_fetch", {"url": "https://example.com/"}),
        text("fair"),
    )

    await story.type("what can reach the web?")
    await story.finished()

    assert any(line.endswith("from net: client_fetch") for line in story.screen()), (
        "a kernel with no engine serves one net tool, and says which"
    )
    assert "no tool is named 'web_search'" in told(story.requests()[2]), (
        "a tool with no engine behind it is not in the table at all: nothing promises it and nothing refuses it"
    )
    assert "web_fetch failed: no tool is named 'web_fetch'" in story.screen(), (
        "and the other half of the group is the same: one name, and nothing behind it"
    )


async def test_a_page_comes_back_as_the_engine_read_it(story_with: Scripted) -> None:
    reader = Reader(
        pages={
            "https://example.com/a": Page(title="An article", content="# Heading\n\nbody"),
            "https://example.com/big": Page(title="A book", content="line\n" * 8000),
        }
    )
    story = await story_with(
        calling("web_fetch", {"url": "https://example.com/a"}),
        calling("web_fetch", {"url": "https://example.com/big"}),
        text("done"),
        engines=paging(reader),
    )

    await story.require("net")
    await story.type("read that page")
    await story.finished()

    assert told(story.requests()[1]) == ("An article — https://example.com/a — 15 bytes markdown\n# Heading\n\nbody"), (
        "the title, the URL it was read from, how much there is, and then the markdown itself"
    )
    assert told(story.requests()[2]).splitlines()[0] == (
        "A book — https://example.com/big — 40.0 KB markdown (kept)"
    ), "a page too long to send is kept, and the line says so beside the handle"
    assert "[d1]" in told(story.requests()[2]), "and the handle is a document, the same drawer a read uses"
    assert reader.asked == ["https://example.com/a", "https://example.com/big"], "one request per call"
    assert story.questions() == (), "a page read through somebody else's network is not this host's question"


def _handle(told_so_far: str) -> str:
    """The handle the previous call handed back, as the next call would carry it."""
    marker = told_so_far.split("[", 1)[1].split("]", 1)[0]
    assert marker.startswith("d"), f"no handle in {told_so_far!r}"
    return marker


def _reasons(story: Story, which: int = 0) -> list[str]:
    """The reasons one question gave, as the sentences a person read: the only place they are observable."""
    return story.questions()[which].prompt.split("? ", 1)[1].split("; ")
