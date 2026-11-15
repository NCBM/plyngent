"""A named endpoint, the dialects it can speak, and the one model that posts to it.

This is the only file in the kernel that opens a socket to a model, and it is deliberately narrow: a
:class:`Route` says *where* and *as whom*, a :class:`~plyngent.kernel.lm.codec.Codec` says *in what
words*, and :class:`Posted` puts the two together — once per attempt, with the retry the failure's kind
implies and the timeout the route was given.

**A route is a name, not a provider.** ``name`` is what a session's setting holds and what ``SetRoute``
takes; ``dialect`` is a word in the table of dialects; nothing here compares either to a company. That is
the whole reason the table is a value the composition root hands in rather than something this file could
have built for itself: which endpoints exist is a fact about how the program was started.

**Retry belongs here and not in a frontend.** ``main`` retried on the CLI's side, which meant a second
consumer would have had to write it again and a cancelled turn had to unwind a sleep somebody else owned.
Here the delays are a route's, one attempt is one request, and the decision "try again" is
:data:`plyngent.kernel.toolresult.OUTCOMES`'s answer for the kind of trouble — the same table a tool's
retry comes from. A sleep inside a turn is cancellable for free, because the whole call is what the turn's
scope races.

**A response that is not an answer is a failure, not a reply.** A status at or above 400 is
:func:`plyngent.kernel.lm.codec.refused`; a body the dialect cannot read is the codec's ``PROTOCOL``. Both
arrive as :class:`~plyngent.kernel.model.ModelFailed`, which is what lets a turn say why it could not run
instead of raising into a task nobody awaits.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast, override

import msgspec
import niquests
from msgspec import Struct

from plyngent.kernel.failure import Failure, Retry
from plyngent.kernel.lm.caps import Caps
from plyngent.kernel.lm.codec import FAILURE_STATUS, refused
from plyngent.kernel.model import Model, ModelFailed
from plyngent.kernel.toolresult import OUTCOMES

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any

    from plyngent.kernel.lm.codec import Codec
    from plyngent.kernel.model import Reply, Request

#: How long one attempt may take when a route does not say. Long on purpose: a model answering a large
#: request legitimately takes minutes, and cutting it off at the length of a web page would break the
#: requests that matter most. What bounds a caller's patience is the turn's own cancellation, not this.
DEFAULT_TIMEOUT = 600.0

#: What a session that has no route at all is told about the world: it can see nothing and carries
#: nothing. Not a refusal — a route that is missing is answered where a route is *named*
#: (:meth:`Routes.resolve`), and what assembling a prompt needs is only "which of the caps apply".
NO_ROUTE_CAPS = Caps(max_output=0)


class Route(Struct, frozen=True, kw_only=True):
    """One endpoint of one model: where it is, what it answers in, and what it can do.

    ``secret`` is the credential and it is a value here rather than something read from the environment:
    what a program is allowed to talk to is a fact about how it was started, and a route that went
    looking would be a second place that decides. The *name* of the variable it came from never reaches
    this field — configuration resolves it, and a route holds only what it will send (§0.11, E71/E77).
    ``attempts`` is the delays between tries — empty is "one try", and the tuple is what a story sets to
    a hundredth of a second so a story about a retry costs no clock.
    """

    name: str = ""
    #: Which row of :data:`plyngent.kernel.lm.DIALECTS` writes and reads this endpoint's JSON.
    dialect: str = ""
    #: Where the endpoint lives, without the path: ``https://host/v1`` and the dialect adds its own.
    base: str = ""
    model: str = ""
    secret: str = ""
    caps: Caps
    #: The header the credential travels in, spelled per route because it is the one thing no compatible
    #: gateway agrees about: a bearer token, an ``x-api-key``, a vendor's own.
    auth: tuple[str, str] = ("Authorization", "Bearer {secret}")
    attempts: tuple[float, ...] = ()
    timeout: float = DEFAULT_TIMEOUT
    headers: tuple[tuple[str, str], ...] = ()

    def codec(self) -> Codec | None:
        """The dialect this route says it speaks, or ``None`` when this build does not speak it.

        The table is read here rather than imported at the top because the table's module imports *this*
        one to file the route type: looked up per call, the two modules cannot get into an argument about
        which of them is imported first.
        """
        from plyngent.kernel.lm import codec_named

        return codec_named(self.dialect)

    def wired(self) -> dict[str, str]:
        """Every header one request carries: the credential, the content type, and anything added."""
        sent = dict(self.headers)
        sent["Content-Type"] = "application/json"
        name, value = self.auth
        if self.secret:
            sent[name] = value.replace("{secret}", self.secret)
        return sent


class Routes:
    """The endpoints this program was started with, and which one answers when nobody says.

    A plain value rather than a component, for the same reason a skill store is one: there is nothing to
    open and nothing to close, so there is no lifecycle a component would have answered. What a route
    table is is a fact about how the program was started, and it is handed to the host by the composition
    root exactly like the model's channel and the skills directory.
    """

    routes: tuple[Route, ...]
    default: str

    def __init__(self, routes: Iterable[Route] = (), *, default: str = "") -> None:
        self.routes = tuple(routes)
        self.default = default

    def named(self, name: str) -> Route | None:
        """The route *name* files, or ``None``: strict, because a name somebody wrote down is a fact.

        ``SetRoute`` answers this one's ``None`` with a refusal, since a person asked for a route by name
        and the name has to mean something.
        """
        for route in self.routes:
            if route.name == name:
                return route
        return None

    def resolve(self, name: str) -> Route | None:
        """Which route answers a session whose setting is *name*: its own, or this host's default.

        The fallback is not laxity, it is what an empty name means — "this session never chose" — and what
        a name this build no longer has means too: a configuration file changed, and a conversation
        should not become unusable because a model was retired. A session is not silently moved in
        silence, either: the turn says which route answered (``Routed`` when it changes, and the attempt
        records the one it used).
        """
        found = self.named(name) if name else None
        if found is not None:
            return found
        return self.named(self.default) if self.default else None

    def caps(self, name: str) -> Caps:
        """What the route answering *name* can do, and the quiet answer when nothing answers it.

        Only for assembling a prompt, which needs one bit — whether pixels may travel — and must not fail
        because a route is missing: refusing a turn is :meth:`resolve`'s caller's business, and it happens
        before any prompt is built.
        """
        found = self.resolve(name)
        return NO_ROUTE_CAPS if found is None else found.caps


class Posted(Model):
    """One route, one request, and as many tries as the kind of trouble allows."""

    _routes: Routes

    def __init__(self, routes: Routes) -> None:
        self._routes = routes

    @override
    async def respond(self, request: Request) -> Reply:
        """Ask the route this request names, once per attempt, and answer what it said.

        The route is resolved from the *request* and not from the model: a session's route is a setting,
        and a setting that changed mid-turn takes effect at the next request — which is the boundary this
        method is called at.
        """
        route = self._routes.resolve(request.route)
        if route is None:
            msg = f"no route answers {request.route or 'this session'}, and this kernel has none by default"
            raise ModelFailed(Failure.MISSING, msg)
        codec = route.codec()
        if codec is None:
            msg = f"route {route.name} names the dialect {route.dialect!r}, which this build does not speak"
            raise ModelFailed(Failure.ARGS, msg)
        body = msgspec.json.encode(codec.encode(request, model=route.model, caps=route.caps))
        return await self._tried(route, codec, body)

    async def _tried(self, route: Route, codec: Codec, body: bytes) -> Reply:
        """One attempt, and another while the kind of trouble is one the world can fix.

        The delays are the route's and the *decision* is the outcome table's: a timeout and a busy server
        are tried again, a wrong key and a wrong request are not. A cancel arriving during the wait stops
        it, because that wait happens inside the turn's scope like everything else.
        """
        failure: ModelFailed | None = None
        for wait in (0.0, *route.attempts):
            if wait:
                await asyncio.sleep(wait)
            try:
                return await self._once(route, codec, body)
            except ModelFailed as failed:
                failure = failed
                if OUTCOMES[failed.failure][1] is not Retry.SAME_CALL:
                    raise
        assert failure is not None, "a loop with at least one attempt either answered or failed"
        raise failure

    async def _once(self, route: Route, codec: Codec, body: bytes) -> Reply:
        """One request to one endpoint, as the answer or as the kind of trouble it met."""
        url = route.base.rstrip("/") + codec.endpoint
        async with niquests.AsyncSession() as session:
            try:
                response = await session.request(
                    "POST",
                    url,
                    headers=route.wired(),
                    data=body,
                    timeout=route.timeout,
                    allow_redirects=False,
                )
            except niquests.RequestException as exc:
                raise _trouble(exc, url=url, seconds=route.timeout) from exc
            sent = await _content(response)
        status = int(getattr(response, "status_code", 0) or 0)
        if status >= FAILURE_STATUS:
            raise refused(codec, sent, status=status, where=route.name)
        return codec.decode(sent)


async def _content(response: Any) -> bytes:
    """The bytes a response carries, whichever way this client hands them over.

    One ``await`` at most: the response of a non-streaming request is already on this side, and a client
    that changes its mind about that should not turn into a second code path in here.
    """
    found = cast("Any", getattr(response, "content", b""))
    if hasattr(found, "__await__"):
        found = await found
    return found if isinstance(found, bytes) else b""


def _trouble(exc: niquests.RequestException, *, url: str, seconds: float) -> ModelFailed:
    """One transport failure as the kind of trouble it is, worded the same way a web tool's is.

    A timeout is its own kind because the same request may well work later; everything else is
    ``PROTOCOL`` and reported as the client said it, which is the wording nobody has to guess at.
    """
    if isinstance(exc, niquests.Timeout):
        return ModelFailed(Failure.TIMEOUT, f"{seconds} seconds passed and {url} had not answered")
    return ModelFailed(Failure.PROTOCOL, f"{url} could not be reached: {exc}")
