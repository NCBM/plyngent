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

from plyngent.kernel.net import bing, deepseek
from plyngent.kernel.net.web import Provider, Providers

#: What this build can do on somebody else's network, by the name a file writes down.
WEB_PROVIDERS = Providers(
    (
        Provider(name=deepseek.NAME, dialects=deepseek.DIALECTS, engine=deepseek.engine),
        Provider(name=bing.NAME, dialects=(), engine=bing.engine, routed=False),
    )
)
