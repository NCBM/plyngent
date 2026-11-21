"""The provider layer: a route, a dialect, and what each can do — and nothing that knows a company.

Everything a model needs beyond the canonical values lives here, and nothing above this package knows
it exists: :mod:`plyngent.kernel.turn` asks a :class:`~plyngent.kernel.model.Model` for a reply, and a
:class:`~plyngent.kernel.lm.route.Posted` turns that question into one HTTP request to the endpoint a
:class:`~plyngent.kernel.lm.route.Route` names. The four pieces are the whole of it:

* :mod:`plyngent.kernel.lm.caps` — one route's model, as the facts anything downstream obeys.
* :mod:`plyngent.kernel.lm.thinking` — the shapes a request may ask a model to think in.
* :mod:`plyngent.kernel.lm.codec` — a dialect as two pure functions, plus the one status→kind table.
* :mod:`plyngent.kernel.lm.route` — a named endpoint, and the model that posts to it.

**One dialect is one file and one row.** That is the acceptance for this whole segment, and this module
is where the rows are: a new provider is a module implementing :class:`~plyngent.kernel.lm.codec.Codec`
and one entry in :data:`DIALECTS`, and no other file changes. The test that holds it is a walk of the
tree looking for a dialect's name *outside* this package — because the day a loop branches on
``if provider == …``, the row was not enough and the shape has to be looked at again.

Two rows may be one class filed twice, when two families of one wire differ in a single word (E167): the
name is still the handle a route carries, and it still says everything the codec needs to know.
"""

from __future__ import annotations

from plyngent.kernel.lm.caps import Caps as Caps
from plyngent.kernel.lm.caps import PrefixCache as PrefixCache
from plyngent.kernel.lm.chat import CHAT as CHAT
from plyngent.kernel.lm.chat import CHAT_LEGACY as CHAT_LEGACY
from plyngent.kernel.lm.chat import Chat as Chat

# Not a type-only import even though only an annotation names it: the package face is what a dialect is
# written against, so ``from plyngent.kernel.lm import Codec`` has to work for a reader of this build.
from plyngent.kernel.lm.codec import Codec as Codec  # noqa: TC001
from plyngent.kernel.lm.messages import MESSAGES as MESSAGES
from plyngent.kernel.lm.messages import Messages as Messages
from plyngent.kernel.lm.responses import RESPONSES as RESPONSES
from plyngent.kernel.lm.responses import Responses as Responses
from plyngent.kernel.lm.route import DEFAULT_TIMEOUT as DEFAULT_TIMEOUT
from plyngent.kernel.lm.route import Posted as Posted
from plyngent.kernel.lm.route import Route as Route
from plyngent.kernel.lm.route import Routes as Routes
from plyngent.kernel.lm.thinking import Adaptive as Adaptive
from plyngent.kernel.lm.thinking import Budget as Budget
from plyngent.kernel.lm.thinking import Effort as Effort
from plyngent.kernel.lm.thinking import Level as Level
from plyngent.kernel.lm.thinking import Off as Off
from plyngent.kernel.lm.thinking import Template as Template
from plyngent.kernel.lm.thinking import ThinkingSpec as ThinkingSpec

#: The dialects this build speaks, one row each. A route names one by its ``name``, and nothing else in
#: the tree names a dialect at all.
DIALECTS: tuple[Codec, ...] = (Messages(), Chat(), Chat(legacy=True), Responses())


def codec_named(name: str) -> Codec | None:
    """The dialect filed under *name*, or ``None`` when this build does not speak it.

    ``None`` and not an exception: a name is a word from a configuration file, and a route naming a
    dialect this build does not have is a fact about the configuration; whether that is a failure belongs
    to the caller, who says which of the two it is (see :class:`~plyngent.kernel.lm.route.Routes`).
    """
    for codec in DIALECTS:
        if codec.name == name:
            return codec
    return None
