"""The web half of the net group: somebody else's network, and the engines this program may ask it with.

A program's web tools are not one backend but a small table of them, and which of them exist is a fact
about how the program was started, and no call decides it. :class:`Engine` is one entry in that table: a
name a person and a model can both say, and up to two verbs, each present only when that engine really
does it.

**Two verbs, and neither is implied by the other.** ``search`` asks an index and answers hits; ``page``
reads one URL and answers a title and readable text. An engine may have one and not the other — Bing
answers queries and cannot render a page — which is why they are two optional verbs on one value rather
than two tables: "may I read a page" is then one scan of one list, and so is "may I search".

**``page`` promises readable form, not a response.** The conversion — HTML to Markdown, a PDF to its text
— happens inside the engine, so the tool hands a model prose and never a body. A channel that can only
move bytes (the local ``client_fetch``) therefore does not implement it, and is never a fallback for it:
"fetch a page" and "read a page" would otherwise quietly become the same tool.

**A search is handed the gate, and a page read is not.** Bing's endpoint is an ordinary HTTP GET to
somebody's server, so it leaves through :mod:`~plyngent.kernel.net.client` and the same question a fetch
asks. A vendor's own endpoint is the one a program was configured with and reads the page on its own
network, so there is no destination of this machine's to agree to. Saying so in the signature keeps the
difference in one place (§0.9).

**The order is the fallback order.** :func:`search_in_order` asks each engine in turn and the first answer
wins, so an engine that is rate-limited costs a sentence in the summary and not the call. It is also the
order a model reads when it is told which names it may pass: not alphabetical, but what the program was
configured with.

**A verb is handed the route the conversation is speaking on** (``on``), because an engine that has more
than one account charges the one being answered on when it can: the bill stays in one place, and nobody
has to name an account — which is the one thing a name must never do.

**An engine arrives by a channel naming it.** :class:`Providers` is what a build ships, and a route's
``provider`` is how one of them is installed: that route's base URL and key become one *account* of that
engine, so two routes naming the same implementation are one engine holding two accounts. An engine with
no account to configure — one that answers a query over somebody's public index —
is in a program only because the program asked for it by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from plyngent.kernel.cancel import Cancelled

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.lm.route import Route
    from plyngent.kernel.net.client import Gate


@dataclass(frozen=True, slots=True)
class Page:
    """A public page as somebody else's network read it: its title, and its content as markdown."""

    title: str = ""
    content: str = ""


@dataclass(frozen=True, slots=True)
class Hit:
    """One page a search matched, as a model reads it: what it is called, where it is, and a line of it."""

    title: str = ""
    url: str = ""
    snippet: str = ""


class EngineFailed(Exception):  # noqa: N818 — an engine that could not answer, not a failure of a call
    """One engine could not answer, in its own words.

    It never becomes a call's failure: :func:`search_in_order` records it and asks the next engine, and
    what is left is a sentence in the summary. That is why it carries prose — that is all it is for.
    """


class SearchVerb(Protocol):
    """One engine's way of answering a query, over a network the caller may have to ask about."""

    async def __call__(self, query: str, *, gate: Gate, on: str = "") -> Sequence[Hit]:
        """Answer *query*, or raise :class:`EngineFailed` saying why not."""
        ...


class PageVerb(Protocol):
    """One engine's way of reading a page through somebody else's network."""

    async def __call__(self, url: str, *, on: str = "") -> Page:
        """Read *url* and answer its title and its content, rendered as text."""
        ...


@dataclass(frozen=True, slots=True)
class Engine:
    """One implementation of the web half: what it is called, and the verbs it can be asked for.

    ``name`` is the word a model passes to ``web_search`` and the word a refusal lists, so it names the
    *implementation* and never an account: which route pays for a search is the engine's own choice
    (see the module docstring), and making it a name a model could say would turn somebody's quota into
    an argument.
    """

    name: str
    search: SearchVerb | None = None
    page: PageVerb | None = None


class Engines:
    """The web half this host was given, in the order a search asks it.

    A plain value, with no component around it, for the same reason a route table is one: there is nothing to
    open and nothing to close, and which engines exist is a fact about how the program was started. What
    a *tool* does with it is decided where the tools are assembled (§0.9, E26).
    """

    engines: tuple[Engine, ...]

    def __init__(self, engines: Sequence[Engine] | None = None) -> None:
        self.engines = () if engines is None else tuple(engines)

    def named(self, name: str) -> Engine | None:
        """The engine called *name*, or ``None``: a name somebody wrote down is a fact, not a hint."""
        for engine in self.engines:
            if engine.name == name:
                return engine
        return None

    def names(self) -> tuple[str, ...]:
        """What this program calls each of its engines, in the order it asks them."""
        return tuple(engine.name for engine in self.engines)

    def asking(self) -> tuple[Engine, ...]:
        """The engines that can answer a query, in the order they are asked."""
        return tuple(engine for engine in self.engines if engine.search is not None)

    def paging(self) -> tuple[Engine, ...]:
        """The engines that can read a page, in the order they are tried."""
        return tuple(engine for engine in self.engines if engine.page is not None)

    def __bool__(self) -> bool:
        return bool(self.engines)

    @override
    def __repr__(self) -> str:
        return f"Engines({', '.join(self.names())})"


#: What an engine asks for when nothing says otherwise: how many hits one search asks for, and how long
#: one request may take. A program's ``[web]`` replaces these, and a route's own ``[route.web]`` may
#: replace them again for the one account that needs it.
DEFAULT_COUNT = 10
DEFAULT_TIMEOUT = 30.0


class EngineFactory(Protocol):
    """How one implementation turns the routes that name it into the engine those routes give."""

    def __call__(self, routes: Sequence[Route], *, count: int, timeout: float) -> Engine | None:
        """The engine these routes make, or ``None`` when none of them can charge it."""
        ...


@dataclass(frozen=True, slots=True)
class Provider:
    """One web implementation this build ships, as something a route and a file can both name.

    ``name`` is what a route's ``provider`` says and what ``[web].engines`` lists — never a vendor's
    endpoint, which is the route's word. ``dialects`` is what a route naming this provider may speak, and
    it is the implementation's answer, and this program does not decide it: a web engine and a wire format
    are two different things, so the one that knows it also knows which wire formats it arrives on.

    ``routed`` is what separates the two kinds of engine: one a route installs by naming it, and one a
    program has because it asked for it by name. Bing is the second — it answers queries and reads
    nothing, and there is no endpoint of a model's to configure it on.
    """

    name: str
    dialects: tuple[str, ...]
    engine: EngineFactory
    routed: bool = True


class Providers:
    """The web implementations this build ships, as one value: a name is looked up, never guessed."""

    providers: tuple[Provider, ...]

    def __init__(self, providers: Sequence[Provider] = ()) -> None:
        self.providers = tuple(providers)

    def named(self, name: str) -> Provider | None:
        """The implementation called *name*, or ``None`` — a name a file wrote down is a fact."""
        for provider in self.providers:
            if provider.name == name:
                return provider
        return None

    def names(self) -> tuple[str, ...]:
        """What this build calls each of them, in the order they were declared."""
        return tuple(provider.name for provider in self.providers)

    def routed(self) -> tuple[Provider, ...]:
        """The ones a route can install, in the order they are collected when nothing says otherwise."""
        return tuple(provider for provider in self.providers if provider.routed)

    @override
    def __repr__(self) -> str:
        return f"Providers({', '.join(self.names())})"


@dataclass(frozen=True, slots=True)
class Choice:
    """One engine a program has, and the accounts it may be asked on, in the order it falls back through.

    ``accounts`` is what a person reads, not what a model is told: route names, so that "which key will
    this spend" is answerable by looking. It is empty for an engine with nothing to configure.
    """

    name: str
    accounts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Web:
    """The web half a program asked for: which engines, in what order, and the numbers they default to.

    A value, with no registry behind it, because it is what a *file* says: ``[web].engines`` is the enable list
    and the order, ``count`` and ``timeout`` are what every account that does not override them gets, and
    the routes each engine may charge are the ones they named. What turns this into engines is the
    composition root's business.
    """

    engines: tuple[Choice, ...] = ()
    count: int = DEFAULT_COUNT
    timeout: float = DEFAULT_TIMEOUT

    def names(self) -> tuple[str, ...]:
        """What this program calls its engines, in the order it asks them."""
        return tuple(one.name for one in self.engines)

    def notice(self) -> str:
        """One line for a person: the engines in order, and the accounts each may spend.

        A model is told the names (they are an argument's value domain); this is the other reader, and what
        it is told is what a search will actually do — ``deepseek: ds-a → ds-b; bing`` is "the first account
        that answers pays, and if neither does, Bing is asked". Empty when the program has no engine.
        """
        said = [f"{one.name}: {' → '.join(one.accounts)}" if one.accounts else one.name for one in self.engines]
        return "; ".join(said)

    def __bool__(self) -> bool:
        return bool(self.engines)


@dataclass(frozen=True, slots=True)
class Found:
    """What a search came back with: one engine's hits, and what the others said about not answering."""

    hits: tuple[Hit, ...] = ()
    failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Read:
    """What a page read came back with: the page, and what the engines before it said."""

    page: Page | None = None
    failures: tuple[str, ...] = ()


async def search_in_order(engines: Sequence[Engine], query: str, *, gate: Gate, on: str = "") -> Found:
    """Ask each engine in turn; the first one that answers wins.

    An engine that raises, or that answers with nothing, is recorded and the next one gets its chance: a
    broken or rate-limited engine never takes the tool down with it, and what it said travels out to the
    summary, and nothing is swallowed. A cancelled turn is not a failing engine, so it is not caught.
    """
    failures: list[str] = []
    for engine in engines:
        assert engine.search is not None, "an engine that cannot search is not asked to"
        try:
            hits = tuple(await engine.search(query, gate=gate, on=on))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — one engine's trouble is that engine's, and the next is tried
            failures.append(f"{engine.name}: {exc}")
            continue
        if hits:
            return Found(hits=hits, failures=tuple(failures))
        failures.append(f"{engine.name}: no results")
    return Found(failures=tuple(failures))


async def page_in_order(engines: Sequence[Engine], url: str, *, on: str = "") -> Read:
    """Read *url* with the first engine that can, or answer with what each one said instead.

    A page read has no "no results" of its own: an engine either renders the page or fails at it, so the
    only thing left to report when nobody could is the reasons, in the order they were met.
    """
    failures: list[str] = []
    for engine in engines:
        assert engine.page is not None, "an engine that cannot read a page is not asked to"
        try:
            return Read(page=await engine.page(url, on=on), failures=tuple(failures))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — one engine's trouble is that engine's, and the next is tried
            failures.append(f"{engine.name}: {exc}")
    return Read(failures=tuple(failures))
