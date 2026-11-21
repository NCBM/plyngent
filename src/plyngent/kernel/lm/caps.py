"""What one route's model can do, as values a loop reads, so that nothing asks who the provider is.

A capability is a fact about *one endpoint of one model*, not about a company: the same vendor serves a
model that reads pictures and one that does not, and a compatible gateway can do anything its author
felt like implementing. So nothing below this module ever asks a provider's name, and a dialect that
needs to know something about the endpoint it is writing for is handed one of these.

Three rules, and they are what makes the table worth its lines:

* **A field is here because somebody obeys it.** ``pictures`` gates a picture at assembly time,
  ``max_output`` becomes the room an answer is given, ``stream`` decides whether a request asks for one,
  ``tools`` decides how many calls one answer may ask for, and ``thinking`` is the shape a request asks
  a model to think in. A cap nobody reads would be a comment with a default value.
* **``max_output`` has no default.** How much room an answer gets is a decision somebody makes, and a
  dialect that must state it (``messages`` does) would otherwise invent one silently.
* **The defaults are the conservative ones.** A route that was told nothing gets a model that cannot
  see, is asked to think in no shape at all, and is asked for one call at a time — and its author says
  so where the route is declared, which keeps the discovery from arriving as a truncation.

**What is *not* here is the model's either.** How a prompt is cached is a fact about the endpoint, not
about the model behind it: the same model behind a second gateway caches differently, so that mechanism
is the route's (:attr:`plyngent.kernel.lm.route.Route.prefix_cache`, E169).
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct

from plyngent.kernel.lm.thinking import Off, ThinkingSpec


class ToolCalls(StrEnum):
    """How many calls one answer may ask for.

    Two words and not a number: what an endpoint does with a batch is a fact about whether it takes one
    at all, and "at most three" is a limit this build would have to enforce rather than a thing the wire
    says. A third word has a place to go here the day somebody needs it.
    """

    #: One call per answer: the model is asked to take them one at a time, which costs rounds.
    ONE = "one"
    #: One answer may ask for several, and the turn runs them together.
    PARALLEL = "parallel"


class Caps(Struct, frozen=True, kw_only=True):
    """One route's model, as the things anything downstream has to know about it."""

    #: Whether a picture reaches the model at all. False is "send the words, not the pixels", which is
    #: what a tool's handle already says in prose. The dialect's own ``pictures`` is the second gate: both
    #: have to be open, and the route's is a claim about the model.
    pictures: bool = False
    #: How many calls one answer may ask for. ``ONE`` asks the model to take them one at a time — which
    #: costs rounds and buys a gateway that would have refused the batch.
    tools: ToolCalls = ToolCalls.ONE
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
    #: Whether an answer may arrive in pieces. It is one bit and not two: a fragment of an *argument* is
    #: not a thing anything here acts on, so a route that streams its tool arguments into one answer is
    #: just a route that streams. False is "ask and wait", which is what a dialect that cannot read a
    #: stream has to be asked for anyway (:attr:`plyngent.kernel.lm.codec.Codec.sse` says which).
    stream: bool = False
