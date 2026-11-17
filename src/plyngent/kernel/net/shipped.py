"""The web implementations *this build* ships, as the one table every name is looked up in.

A name in a file — a route's ``provider``, an entry of ``[web].engines`` — is looked up here or refused, and
nothing downstream compares it to a company: the kernel's tools read a table of engines, and what fills it
is this file. It lives in this package because this is where the implementations are, so the composition
root can install what a configuration named and the configuration surface can refuse a pair that cannot
exist, without either of them knowing what a vendor is called.

Two kinds of entry, and the difference is ``routed``: ``deepseek`` is installed by a route naming it (that
is where its base URL and its key come from), and ``bing`` has no endpoint to name and is installed by a
program asking for it by that name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.net import bing, deepseek
from plyngent.kernel.net.web import DEFAULT_COUNT, DEFAULT_TIMEOUT, Choice, Provider, Providers, Web

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.kernel.lm.route import Route

#: What this build can do on somebody else's network, by the name a file writes down.
WEB_PROVIDERS = Providers(
    (
        Provider(name=deepseek.NAME, dialects=deepseek.DIALECTS, engine=deepseek.engine),
        Provider(name=bing.NAME, dialects=(), engine=bing.engine, routed=False),
    )
)


def accounts_of(name: str, routes: Sequence[Route]) -> tuple[str, ...]:
    """The routes an engine called *name* may be asked on, in the order it falls back through them.

    The *shared* ones only: a route that shares nothing is still charged for the conversations speaking on
    it, and that is the engine's own first step rather than a fact for this table. What is listed here is
    the fallback a conversation speaking somewhere else gets, which is what a person reading the line wants
    to know.
    """
    return tuple(one.name for one in routes if one.provider == name and one.web is not None and one.web.share)


def web_half(
    routes: Sequence[Route],
    *,
    engines: Sequence[str] | None = None,
    count: int = DEFAULT_COUNT,
    timeout: float = DEFAULT_TIMEOUT,
) -> Web:
    """The web half a program has: the implementations its routes install, or exactly the ones it named.

    ``engines`` is ``[web].engines`` — an enable list *and* an order, because a rule that only sorted would
    leave "is this engine on at all" with two answers: the shared accounts decide whose key pays, and this
    decides whether the engine is there. ``None`` is a program that said nothing, and what it gets is every
    routed implementation some route names — installing one is what naming it in ``provider`` means, and a
    route sharing nothing is still a program with web tools. An engine no route installs is never collected:
    it has no endpoint to collect it from, so it is in a program only when this list names it.
    """
    named = (
        tuple(engines)
        if engines is not None
        else tuple(
            provider.name
            for provider in WEB_PROVIDERS.routed()
            if any(route.provider == provider.name for route in routes)
        )
    )
    return Web(
        engines=tuple(Choice(name=name, accounts=accounts_of(name, routes)) for name in named),
        count=count,
        timeout=timeout,
    )
