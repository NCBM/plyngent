"""What one route's model can do, as values a loop reads, so that nothing asks who the provider is.

A capability is a fact about *one endpoint of one model*, not about a company: the same vendor serves a
model that reads pictures and one that does not, and a compatible gateway can do anything its author
felt like implementing. So nothing below this module ever asks a provider's name, and a dialect that
needs to know something about the endpoint it is writing for is handed one of these.

Three rules, and they are what makes the table worth its lines:

* **A field is here because somebody obeys it.** ``vision`` gates a picture at assembly time,
  ``prefix_cache`` decides which cache mechanism a request carries (§10.5), ``max_output`` becomes the
  room an answer is given, ``stream`` decides whether a request asks for one, ``parallel_tools`` becomes
  a flag the dialects write, and ``thinking`` is the shape a request asks a model to think in. A cap
  nobody reads would be a comment with a default value.
* **``max_output`` has no default.** How much room an answer gets is a decision somebody makes, and a
  dialect that must state it (``messages`` does) would otherwise invent one silently.
* **The defaults are the conservative ones.** A route that was told nothing gets a model that cannot
  see, is asked to think in no shape at all, and is asked for one call at a time — and its author says
  so where the route is declared, which keeps the discovery from arriving as a truncation.
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct

from plyngent.kernel.lm.thinking import Off, ThinkingSpec


class PrefixCache(StrEnum):
    """How this route is told to reuse the front of a prompt, in the three mechanisms there are.

    The distinction is not cosmetic: it decides *which field a request carries*, and a provider handed
    a mechanism it does not implement either ignores it (a miss nobody accounts for) or refuses the
    request. §10.5 spells out why each one fails the way it does.
    """

    #: Nothing is said: the gateway's own implicit behaviour, or none at all.
    NONE = "none"
    #: A short stable key names the prefix (``prompt_cache_key``); the provider hashes what follows.
    IMPLICIT = "implicit"
    #: A marker inside the request says "everything before this never moves" (``cache_control``).
    BREAKPOINT = "breakpoint"


class Caps(Struct, frozen=True, kw_only=True):
    """One route's model, as the things anything downstream has to know about it."""

    #: Whether a picture reaches the model at all. False is "send the words, not the pixels", which is
    #: what a tool's handle already says in prose.
    vision: bool = False
    #: Whether one answer may ask for more than one call. False asks the model to take them one at a
    #: time — which costs rounds and buys a gateway that would have refused the batch.
    parallel_tools: bool = False
    #: Whether this model is asked to think, and in which of the shapes the wires have for it. The shape
    #: is the fact a request carries, and which shapes a dialect can write is that dialect's declaration
    #: (:attr:`plyngent.kernel.lm.codec.Codec.spells`, E168).
    thinking: ThinkingSpec = Off()
    #: How many tokens an answer may take. Not defaulted: it is a budget somebody picks, and a dialect
    #: that must state it would otherwise pick for them.
    max_output: int
    #: How many tokens this model can hold at once, counting what it was asked and what it answers. Zero
    #: is "this route never said" — a third-party endpoint's window is a fact its provider knows and this
    #: file might not, and a denominator nobody stated must not read as a small one. Nothing a request
    #: carries depends on it: it is the number a frontend divides by when it says how full the context is.
    context_window: int = 0
    #: Which cache mechanism this route's requests carry.
    prefix_cache: PrefixCache = PrefixCache.NONE
    #: Whether an answer may arrive in pieces. It is one bit and not two: a fragment of an *argument* is
    #: not a thing anything here acts on, so a route that streams its tool arguments into one answer is
    #: just a route that streams. False is "ask and wait", which is what a dialect that cannot read a
    #: stream has to be asked for anyway (:attr:`plyngent.kernel.lm.codec.Codec.sse` says which).
    stream: bool = False
