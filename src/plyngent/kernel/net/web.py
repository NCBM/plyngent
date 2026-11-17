"""The web half of the net group: somebody else's network, and the engines this program may ask it with.

A program's web tools are not one backend but a small table of them, and which of them exist is a fact
about how the program was started rather than about a call. :class:`Engine` is one entry in that table: a
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from plyngent.kernel.cancel import Cancelled

if TYPE_CHECKING:
    from collections.abc import Sequence

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

    A plain value rather than a component, for the same reason a route table is one: there is nothing to
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
    summary instead of being swallowed. A cancelled turn is not a failing engine, so it is not caught.
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
