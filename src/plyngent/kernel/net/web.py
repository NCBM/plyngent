"""The vendor's half of the net group: a page read elsewhere, and a search asked in order.

Two capabilities and one value, because they are one backend from the composition root's point of view:
``boot.install(web=…)`` is something a program either has or has not, and which tools that enables is
decided from what it holds (§0.9, E26). The two are separate protocols rather than one, because a
backend may be able to answer a query and not read a page, and a tool that only needs the first should
not be handed the second.

:class:`WebChannel` is a component for the same reason :class:`~plyngent.kernel.model.ModelChannel` is:
installing a backend is a declaration and swapping it is a ``replace``, and "nothing installed" is a
value (``None``) rather than a field nobody set — so a kernel with no backend refuses to offer the tools
instead of failing at their first call.

**A source is handed the gate.** Bing's endpoint is an ordinary HTTP GET to somebody's server, so it goes
through the same engine and the same question a fetch does; the vendor's own API does not, because that
request goes to the endpoint a program was configured with and the page is fetched on the vendor's
network. Saying that in the signature keeps the difference in one place (§0.9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from composat import Component

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


class SourceFailed(Exception):  # noqa: N818 — a source that could not answer, not a failure of a call
    """One search source could not answer, in its own words.

    It never becomes a call's failure: :func:`search_in_order` records it and asks the next source, and
    what is left is a sentence in the summary. That is why it carries prose — that is all it is for.
    """


class PageSource(Protocol):
    """One way to read a page through somebody else's network."""

    async def fetch(self, url: str) -> Page:
        """Read *url* and answer its title and content."""
        ...


class SearchSource(Protocol):
    """One way to answer a query, over a network the caller may have to ask about."""

    async def search(self, query: str, *, gate: Gate) -> Sequence[Hit]:
        """Answer *query*, or raise :class:`SourceFailed` saying why not."""
        ...


@dataclass(frozen=True, slots=True)
class Web:
    """What the composition root installed: one page reader, and the search sources in fallback order.

    Both empty is a legal value and means "no web tools": a program that configured no key has no search
    and no page read, and says so by the two tools not being in the table at all.
    """

    pages: PageSource | None = None
    sources: tuple[SearchSource, ...] = ()


class WebChannel(Component):
    """The web tools this host has, or nothing — a component, so that installing one is a declaration."""

    inner: Web | None

    def __init__(self, inner: Web | None = None) -> None:
        super().__init__()
        self.inner = inner

    @override
    def __repr__(self) -> str:
        return f"WebChannel({self.inner!r})"

    @property
    def installed(self) -> bool:
        """Whether a backend is there at all."""
        return self.inner is not None

    @property
    def pages(self) -> PageSource | None:
        """The page reader, when one is installed."""
        return None if self.inner is None else self.inner.pages

    @property
    def sources(self) -> tuple[SearchSource, ...]:
        """The search sources, in the order they are asked."""
        return () if self.inner is None else self.inner.sources


@dataclass(frozen=True, slots=True)
class Found:
    """What a search came back with: one source's hits, and what the others said about not answering."""

    hits: tuple[Hit, ...] = ()
    failures: tuple[str, ...] = ()


async def search_in_order(sources: Sequence[SearchSource], query: str, *, gate: Gate) -> Found:
    """Ask each source in turn; the first one that answers wins.

    A source that raises, or that answers with nothing, is recorded and the next one gets its chance: a
    broken or rate-limited source never takes the tool down with it, and what it said travels out to the
    summary instead of being swallowed. A cancelled turn is not a failing source, so it is not caught.
    """
    failures: list[str] = []
    for source in sources:
        name = type(source).__name__
        try:
            hits = tuple(await source.search(query, gate=gate))
        except Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001 — one source's trouble is that source's, and the next one is tried
            failures.append(f"{name}: {exc}")
            continue
        if hits:
            return Found(hits=hits, failures=tuple(failures))
        failures.append(f"{name}: no results")
    return Found(failures=tuple(failures))
