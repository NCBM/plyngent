"""The net group's own tool: ``client_fetch``, a request that leaves from *this* machine.

The engine (:mod:`plyngent.kernel.net.client`) makes the request and the policy
(:mod:`plyngent.kernel.net.policy`) says what a URL is; what is left for this module is the part only a
tool can do — read a call's arguments, put the question to a person, and shape the answer for a model.

**The question is here, and the policy only says whether there is one.** A private address, a cleartext
transport, a verb that changes something and a body that leaves the machine are four reasons, and three
kinds; :meth:`NetPolicy.needs` reports which ones this request trips and never asks anybody, this module
asks once with all of them in the same sentence (§0.9, E34). What the answer is worth is kept in the
same place: a read that was not agreed to may be tried again, a write that was not agreed to may not
(E33) — the difference lives in :func:`_unagreed` and nowhere else.

**A body too long for the conversation is not cut.** It goes to the document store and the result carries
the handle, exactly as a read of a long file does (E28): one drawer, one resumption
(``read(document_id=…)``), and no second way to page through what a tool produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import msgspec

from plyngent.kernel.documents import MAX_KEPT_BYTES, picture_media
from plyngent.kernel.failure import Failure
from plyngent.kernel.net import client, policy, web
from plyngent.kernel.toolresult import DenialKind, Effects, Hint, Part, Text, ToolResult, kind_of
from plyngent.kernel.tools.document import kept, kept_bytes
from plyngent.kernel.toolspec import Group, Mutation, Res, ToolArgs, ToolSpec
from plyngent.wire.ask import Agreed, By

if TYPE_CHECKING:
    from collections.abc import Callable

    from plyngent.kernel.env import Env

#: How many bytes a size is written in, at the two steps the summary reads in: ``1.2 KB``, ``3.4 MB``.
_KILO = 1000
_MEGA = 1_000_000

#: How many hits a search shows by default: enough to choose from, and not a page of links.
DEFAULT_MAX_HITS = 8


class ClientFetchArgs(ToolArgs, kw_only=True):
    """Send one HTTP request from this machine, and read what came back.

    Use it for public docs and APIs, and for a server on this machine or the local network once you have
    agreed to reach it. The request leaves from *this* host, so what can be reached is what this host can
    reach: a loopback or private address is exactly the case the person is asked about.

    ``method`` is one of DELETE, GET, HEAD, POST, PUT. ``headers`` are the request's own, except the ones
    that change what a request *is* rather than what it asks for — the framing and proxy headers are
    refused, and ``user_agent`` is the one identity a call may state, winning over a ``User-Agent`` in
    ``headers``.

    A response body longer than the conversation can hold is kept whole and the result carries a handle:
    ``read(document_id=…)`` resumes it without a second request. ``max_bytes`` caps the download and a cut
    body says ``truncated``; ``max_redirects`` bounds the chain; a redirect that leaves https for plain
    http is refused unless ``allow_http_downgrade=true``.

    A private host is asked about once per process and then remembered by host and port. Plain http, a
    method that changes something on the other end, and a body that leaves this machine are asked about
    every time, because each of them is a fact about this request and not about that machine.
    """

    url: Annotated[str, msgspec.Meta(min_length=1)]
    # Written in name order: msgspec emits the enum in the order the literal spells it (E32).
    method: policy.Method = "GET"
    headers: dict[str, str] | None = None
    body: str | None = None
    user_agent: str | None = None
    timeout_seconds: Annotated[float, msgspec.Meta(gt=0)] = policy.DEFAULT_TIMEOUT_SECONDS
    max_bytes: Annotated[int, msgspec.Meta(ge=1, le=MAX_KEPT_BYTES)] = policy.DEFAULT_MAX_BYTES
    max_redirects: Annotated[int, msgspec.Meta(ge=0)] = policy.DEFAULT_MAX_REDIRECTS
    follow_redirects: bool = True
    allow_http_downgrade: bool = False


async def client_fetch(env: Env, args: ClientFetchArgs) -> ToolResult:
    """Read the call into a request, make it, and shape what came back."""
    request = _read(args)
    if isinstance(request, policy.Unreadable):
        return ToolResult.of(Failure.ARGS, request.message)
    questions = _Questions(env)
    outcome = await client.fetch(request, gate=questions)
    if isinstance(outcome, client.Stopped):
        if outcome.denial is not None:
            return ToolResult.denied(
                outcome.denial,
                outcome.message,
                hints=() if outcome.fix is None else (_hint(outcome.fix),),
                effects=Effects(approvals=tuple(questions.approvals)),
            )
        return ToolResult.of(
            outcome.failure,
            outcome.message,
            hints=() if outcome.fix is None else (_hint(outcome.fix),),
            effects=Effects(approvals=tuple(questions.approvals)),
        )
    return await _answered(env, outcome, approvals=tuple(questions.approvals))


def _read(args: ClientFetchArgs) -> client.Request | policy.Unreadable:
    """The call's arguments as one request, or which argument could not be read and why."""
    url = policy.parse_url(args.url)
    if isinstance(url, policy.Unreadable):
        return url
    method = policy.parse_method(args.method)
    if isinstance(method, policy.Unreadable):
        return method
    headers = policy.request_headers(args.headers, user_agent=args.user_agent)
    if isinstance(headers, policy.Unreadable):
        return headers
    body: bytes | None = None
    if args.body is not None:
        if len(args.body) > policy.MAX_BODY_CHARS:
            return policy.Unreadable(
                "body", f"a request body may be at most {policy.MAX_BODY_CHARS} characters ({len(args.body)} given)"
            )
        body = args.body.encode("utf-8")
    return client.Request(
        method=method,
        url=url.url,
        headers=headers,
        body=body,
        timeout_seconds=args.timeout_seconds,
        max_bytes=args.max_bytes,
        max_redirects=args.max_redirects,
        follow_redirects=args.follow_redirects,
        allow_http_downgrade=args.allow_http_downgrade,
    )


class _Questions:
    """One call's gate: whether this hop may be made, and what a person has already agreed to.

    Per hop rather than per request, because a redirect is a second destination — and per *call* for the
    agreements, because what a person agreed to is this request and the record of it belongs to the
    result this call produces.
    """

    env: Env
    approvals: list[str]

    def __init__(self, env: Env) -> None:
        self.env = env
        self.approvals = []

    async def __call__(self, leg: client.Leg) -> client.Stopped | None:
        """Ask about one hop: nobody, when it needs no agreement; a stop, when it may not be made."""
        assessment = await policy.assess(leg.url, assume_public=self.env.net.assume_public)
        if assessment is None:
            return client.Stopped(Failure.MISSING, f"{leg.url.host} does not resolve to an address")
        if assessment.kind is policy.HostClass.FORBIDDEN:
            addresses = ", ".join(assessment.addresses)
            return client.Stopped(
                Failure.DENIED,
                f"{leg.url.host} resolves to {addresses}, which this build never contacts",
                denial=DenialKind.NEVER_ALLOWED,
            )
        reasons = self.env.net.needs(assessment, method=leg.method, has_body=leg.has_body)
        if not reasons:
            return None
        agreed = await self.env.human.confirm(
            prompt=_asked(leg, reasons), severity=policy.weight(reasons), site=self.env.site("gate")
        )
        if not agreed.accepted:
            return _unagreed(leg, agreed)
        if any(kind is policy.ReasonKind.PRIVATE_HOST for kind, _ in reasons) and agreed.by is not By.POLICY:
            # The one reason that is about the machine rather than about this request, so agreeing to it is
            # agreeing to that host and port and it is not asked about again. Cleartext and a payload are
            # facts about *this* request: remembering one of them would agree to a request nobody described.
            # A policy's yes is not one of these either: it is what the program was started to do, so it
            # grants nothing for later (§0.15, the fourth commit).
            self.approvals.append(self.env.net.approve(leg.url))
        return None


def _asked(leg: client.Leg, reasons: tuple[policy.Reason, ...]) -> str:
    """The one question a request is asked, with every reason it tripped in the same sentence."""
    return f"may I send {leg.method} to {leg.url.url}? {'; '.join(text for _kind, text in reasons)}"


def _unagreed(leg: client.Leg, answer: Agreed) -> client.Stopped:
    """What a request that was not agreed to is worth, given who did not agree (§0.15, E118).

    A person saying no — or a policy the program was started with — is a decision, and it is the same
    decision for a read and a write: asking again asks them again. Silence is not a decision, so it is
    read by what the request would cost to repeat: a read is idempotent, so nobody answering is the
    world being unavailable and the same call a minute later may well go through; a write that nobody
    agreed to is not replayed, because it would be the same unagreed request and a model told "try
    again" would ask the same question again.
    """
    message = f"nobody agreed to contact {policy.grant_key(leg.url.host, leg.url.port)}, so the request was not sent"
    if answer.by in {By.HUMAN, By.POLICY} or leg.method not in policy.READ_ONLY:
        return client.Stopped(Failure.DENIED, message, denial=kind_of(answer))
    return client.Stopped(Failure.BUSY, message)


async def _answered(env: Env, fetched: client.Fetched, *, approvals: tuple[str, ...]) -> ToolResult:
    """The result: one line that says everything but the body, and the body as text or as a handle."""
    effects = Effects(
        external=(f"{fetched.status} {fetched.method} {fetched.url.url}",),
        approvals=approvals,
    )
    if fetched.kind is client.BodyKind.EMPTY:
        return ToolResult.ok(_summary(fetched, kept=False), effects=effects)
    if fetched.kind is client.BodyKind.TEXT:
        text = await _text(env, fetched.as_text(), line=lambda kept: _summary(fetched, kept=kept))
        if isinstance(text, ToolResult):
            return text
        body, summary = text
        return ToolResult.ok(summary, body=body, effects=effects)
    media = picture_media(fetched.body) or _media_of(fetched)
    summary = _summary(fetched, kept=True)
    picture = await kept_bytes(env, fetched.body, summary=summary, media=media)
    if isinstance(picture, ToolResult):
        return picture
    return ToolResult.ok(summary, body=picture, effects=effects)


async def _text(env: Env, content: str, *, line: Callable[[bool], str]) -> tuple[Part, str] | ToolResult:
    """A body as the model gets it, and the line that stands in for it when it did not fit.

    ``line`` is given whether the body had to be kept, because the handle's own summary has to say that it
    *is* a handle — and the line beside the handle is the one the model reads.
    """
    long = len(content) > env.documents.keep_chars
    summary = line(long)
    part = await kept(env, content, summary=summary)
    return part if isinstance(part, ToolResult) else (part, summary)


def _summary(fetched: client.Fetched, *, kept: bool) -> str:
    """Everything about the answer except the answer: the status, the line, the size and what became of it."""
    notes = list(fetched.notes)
    if fetched.truncated:
        notes.append("truncated")
    if kept:
        notes.append("kept")
    if fetched.redirects:
        notes.append(f"{fetched.redirects} {_redirects(fetched.redirects)}")
    beside = f" ({', '.join(notes)})" if notes else ""
    return f"{fetched.status} {fetched.method} {fetched.url.url} — {_size(len(fetched.body))} {fetched.kind}{beside}"


def _hint(fix: client.Fix) -> Hint:
    """The argument that would have let this call finish, as the call it belongs to."""
    match fix:
        case client.Fix.ALLOW_DOWNGRADE:
            return Hint(
                tool=CLIENT_FETCH.name,
                given=(("allow_http_downgrade", "true"),),
                why="the redirect leaves https, which is this call's decision to follow",
            )
        case client.Fix.MORE_REDIRECTS:
            return Hint(
                tool=CLIENT_FETCH.name,
                given=(("max_redirects", "10"),),
                why="the chain is longer than this call allowed",
            )


def _size(count: int) -> str:
    """How big a body is, in the unit a person reads without counting digits."""
    if count < _KILO:
        return f"{count} byte" if count == 1 else f"{count} bytes"
    if count < _MEGA:
        return f"{count / _KILO:.1f} KB"
    return f"{count / _MEGA:.1f} MB"


def _redirects(count: int) -> str:
    """The noun a redirect count is written in, so that one of them does not read like a bug."""
    return "redirect" if count == 1 else "redirects"


def _media_of(fetched: client.Fetched) -> str:
    """The response's content type as a store's media, or the one that says "bytes, not text"."""
    return fetched.content_type.split(";")[0].strip() or "application/octet-stream"


class WebSearchArgs(ToolArgs, kw_only=True):
    """Search the web through the engines this program was built with, and read the pages it names.

    One entry per hit — a title, a URL, and the line the engine had for it — and nothing else: a search
    result is a place to read, not the reading. Use ``client_fetch`` for a page on this machine or a
    local network, and note that a search that reaches a server of its own goes out from here too, so the
    same question about a destination applies to it.

    ``engine`` names *which implementation* to ask — the names are the ones the prompt lists for this
    program — and leaving it out asks them in the order it lists them, falling through to the next when
    one cannot answer. It never names an account or a key: which of an engine's routes pays for a search
    is that engine's own choice, and a search is not a place to argue about somebody's quota.
    """

    query: Annotated[str, msgspec.Meta(min_length=1)]
    engine: str = ""
    max_hits: Annotated[int, msgspec.Meta(ge=1)] = DEFAULT_MAX_HITS


async def web_search(env: Env, args: WebSearchArgs) -> ToolResult:
    """Ask the engine the call named — or each of them in turn — and show the pages that answered."""
    chosen = env.engines.asking()
    if args.engine:
        picked = env.engines.named(args.engine)
        if picked is None or picked.search is None:
            return ToolResult.of(Failure.ARGS, _nobody(args.engine, chosen))
        chosen = (picked,)
    questions = _Questions(env)
    found = await web.search_in_order(chosen, args.query, gate=questions, on=env.route)
    shown = found.hits[: args.max_hits]
    effects = Effects(approvals=tuple(questions.approvals))
    if not shown:
        return ToolResult.ok(f"no hit for {args.query!r} ({_because(found)})", effects=effects)
    body = Text(value="\n".join(_entry(hit) for hit in shown))
    return ToolResult.ok(_found(found, shown, args.query), body=body, effects=effects)


def _nobody(named: str, asking: tuple[web.Engine, ...]) -> str:
    """Why a search cannot ask the engine it named, and which ones it could have asked instead."""
    known = ", ".join(engine.name for engine in asking)
    if not asking:
        return f"no engine can search, so {named!r} is not one of them"
    if named not in {engine.name for engine in asking}:
        return f"no engine is named {named!r}; this program searches with {known}"
    return f"engine {named!r} cannot search; this program searches with {known}"


def _found(found: web.Found, shown: tuple[web.Hit, ...], query: str) -> str:
    """How many of what was found is being shown, and what the engines that did not answer said."""
    total = len(found.hits)
    count = f"{len(shown)} of {total}" if len(shown) < total else str(total)
    said = f"{count} {'hit' if total == 1 else 'hits'} for {query!r}"
    return f"{said} ({_because(found)})" if found.failures else said


def _because(found: web.Found) -> str:
    """Why a search came back with nothing, as what each engine said in its own words."""
    return "; ".join(found.failures) or "no engine is installed"


def _entry(hit: web.Hit) -> str:
    """One hit as the model reads it: what it is called and where it is, then the line the engine had."""
    said = f"- {hit.title or '(untitled)'} — {hit.url}"
    return f"{said}\n  {hit.snippet}" if hit.snippet else said


class WebFetchArgs(ToolArgs, kw_only=True):
    """Read one public page as Markdown, through the network of the engines this program was given.

    Use it for an article or a documentation page when clean text is enough. The page is fetched and
    rendered on somebody else's network, so a target on this machine or a local network is unreachable
    here — ``client_fetch`` is the one that goes out from here, and the one to use for a non-GET method,
    a custom header, or the raw status of a response. An engine that can only move bytes is not one of
    these: what comes back is the page as readable text, never a body to render by hand.

    ``http`` and ``https`` are the only schemes, and a URL with a username or a password in it is refused.
    A page longer than the conversation can hold is kept whole and the result carries a handle:
    ``read(document_id=…)`` resumes it with no second request.
    """

    url: Annotated[str, msgspec.Meta(min_length=1)]


async def web_fetch(env: Env, args: WebFetchArgs) -> ToolResult:
    """Read the page with the first engine that can, and hand the model its title and its text."""
    url = policy.parse_url(args.url)
    if isinstance(url, policy.Unreadable):
        return ToolResult.of(Failure.ARGS, url.message)
    read = await web.page_in_order(env.engines.paging(), url.url, on=env.route)
    if read.page is None:
        return ToolResult.of(Failure.PROTOCOL, _unread(read))
    page = read.page
    found = await _text(env, page.content, line=lambda kept: _page(page, url, kept=kept))
    if isinstance(found, ToolResult):
        return found
    body, summary = found
    return ToolResult.ok(summary, body=body, effects=Effects(external=(f"read {url.url}",)))


def _unread(read: web.Read) -> str:
    """Why nobody read the page, as what each engine said before giving up."""
    return "; ".join(read.failures) or "no engine reads pages"


def _page(page: web.Page, url: policy.ParsedUrl, *, kept: bool) -> str:
    """What an engine read, in one line: what it is called, where it was, and how much of it there is."""
    beside = " (kept)" if kept else ""
    return f"{page.title or '(untitled)'} — {url.url} — {_size(len(page.content))} markdown{beside}"


CLIENT_FETCH = ToolSpec(
    name="client_fetch",
    group=Group.NET,
    args=ClientFetchArgs,
    run=client_fetch,
    # ``HUMAN`` is the question this tool asks for itself; ``NET`` is what it reaches; ``DOCUMENT`` is
    # where a body too long to send goes. What a request may reach is decided by the policy at the moment
    # it is made, so the posture derived from these is ``AUTO`` — the executor must not ask a second time
    # about what the handler already asked (§0.9).
    uses=Res.NET | Res.DOCUMENT | Res.HUMAN,
    # It leaves this machine, and a write leaves it changed somewhere nobody here can see. Declared so
    # that every call has to say why it is being made (§10.10).
    mutates=Mutation.OUTSIDE,
)

WEB_SEARCH = ToolSpec(
    name="web_search",
    group=Group.NET,
    args=WebSearchArgs,
    run=web_search,
    # ``NET`` and not ``HUMAN``: it may have to ask about a destination (an engine that reaches a server
    # of its own does), but the question is not what this tool *is* — a search with nothing to ask runs.
    uses=Res.NET,
    # It reads somebody else's index and changes nothing on either side, so the posture derived from this
    # is ``AUTO``: a search is not a thing the executor has an opinion about.
    mutates=Mutation.NONE,
)

WEB_FETCH = ToolSpec(
    name="web_fetch",
    group=Group.NET,
    args=WebFetchArgs,
    run=web_fetch,
    # ``NET`` for the engines it may ask, ``DOCUMENT`` for a page too long to send. No ``HUMAN``: the
    # request leaves through somebody else's network, so there is no destination of this machine's to
    # agree to.
    uses=Res.NET | Res.DOCUMENT,
    # A page read changes nothing on either side. What it costs is the engine's business, and a charge is
    # not a change to anything here.
    mutates=Mutation.NONE,
)

#: The net tools a kernel with no engine installed serves (§0.9, E26).
SPECS: tuple[ToolSpec, ...] = (CLIENT_FETCH,)

#: The net tools an engine enables; which of them a kernel has is :func:`net_specs`'s answer.
CONDITIONAL: tuple[ToolSpec, ...] = (WEB_SEARCH, WEB_FETCH)

#: Every tool in this module, installed or not: what a structural check walks and a table may hold.
ALL: tuple[ToolSpec, ...] = (*SPECS, *CONDITIONAL)


def net_specs(engines: web.Engines | None) -> tuple[ToolSpec, ...]:
    """The net tools a kernel serves **because** of what was installed, and nothing when nothing was.

    The one place where "does this tool exist" is decided, and it is decided by what the composition root
    holds rather than by what a call names: a kernel with no engine is ``BUILTIN`` alone, and a tool that
    is not there is answered as a name nothing has rather than as a refusal. What never changes with the
    answer is a tool's name and its schema, so a provider's cached prefix stays a function of the *set* a
    session may reach (§10.5).

    Each verb enables its own tool, and nothing else does: an engine that only searches gives a program
    ``web_search`` and no ``web_fetch``, and one that only reads pages gives the other. Which engines are
    in the table at all — one route's ``share`` rather than another's — is decided where the table is
    built, so this stays a question about abilities.
    """
    if engines is None:
        return ()
    served: list[ToolSpec] = []
    if engines.paging():
        served.append(WEB_FETCH)
    if engines.asking():
        served.append(WEB_SEARCH)
    return tuple(served)
