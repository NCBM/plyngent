"""What one route's model can do, as values a loop reads instead of asking who the provider is.

A capability is a fact about *one endpoint of one model*, not about a company: the same vendor serves a
model that reads pictures and one that does not, and a compatible gateway can do anything its author
felt like implementing. So nothing below this module ever asks a provider's name, and a dialect that
needs to know something about the endpoint it is writing for is handed one of these.

Three rules, and they are what makes the table worth its lines:

* **A field is here because somebody obeys it.** ``vision`` gates a picture at assembly time,
  ``prefix_cache`` decides which cache mechanism a request carries (§10.5), ``max_output`` becomes the
  room an answer is given, and ``parallel_tools`` / ``reasoning`` become flags the dialects write.
  A cap nobody reads would be a comment with a default value.
* **``max_output`` has no default.** How much room an answer gets is a decision somebody makes, and a
  dialect that must state it (``messages`` does) would otherwise invent one silently.
* **The defaults are the conservative ones.** A route that was told nothing gets a model that cannot
  see, cannot think out loud, and is asked for one call at a time — and its author says so where the
  route is declared rather than discovering it as a truncation.
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct


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
    """One route's model, as the five things anything downstream has to know about it."""

    #: Whether a picture reaches the model at all. False is "send the words, not the pixels", which is
    #: what a tool's handle already says in prose.
    vision: bool = False
    #: Whether one answer may ask for more than one call. False asks the model to take them one at a
    #: time — which costs rounds and buys a gateway that would have refused the batch.
    parallel_tools: bool = False
    #: Whether this model takes a thinking setting at all, and how many tokens it may spend when it does.
    #: The pair is deliberate: a dialect that wants a flag reads the first, one that wants a number
    #: reads the second, and neither has to know the other exists.
    reasoning: bool = False
    reasoning_budget: int = 0
    #: How many tokens an answer may take. Not defaulted: it is a budget somebody picks, and a dialect
    #: that must state it would otherwise pick for them.
    max_output: int
    #: Which cache mechanism this route's requests carry.
    prefix_cache: PrefixCache = PrefixCache.NONE
