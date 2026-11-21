"""How a model is asked to think, as the shapes the wires have for it.

There is no one field for "think harder". The three dialects in this package spell it three ways, and two
vendors on one wire spell it two more, so what a route can say is a *shape* and not a number: a token
budget, the model's own choice, an effort level, or the chat template's switch. Each shape is one small
value here, and which of them a dialect can write is that dialect's own declaration
(:attr:`plyngent.kernel.lm.codec.Codec.spells`) — so a route asking for a shape its endpoint cannot be
told is refused where the file is read, and not with a 400 on the first turn (§10.4, E168).

**Off is a shape and the default.** It says the request carries nothing at all, which is not the same as
asking for no thinking: a model that thinks by default keeps doing so, and a route that wants that
stopped has no word here to stop it with.

**A level above high is the route's claim about its endpoint.** ``xhigh`` is written verbatim, and an
endpoint that only knows three levels answers ``400``: writing it is saying "mine knows this word", which
is a fact the file's author has and this build does not.
"""

from __future__ import annotations

from enum import StrEnum

from msgspec import Struct


class Level(StrEnum):
    """The effort levels this build can write, as a closed set rather than a free string.

    Closed because a level is a word an endpoint either knows or refuses, and a set this build spells is
    the only one it can promise to spell the same way twice.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    #: Above ``high``, and only an endpoint that says so takes it; see this module's docstring.
    XHIGH = "xhigh"


#: The word each shape goes by, in a configuration file and in a sentence about a dialect. One constant
#: per shape, because the tag a file is read by is also what a refusal says, and two spellings drift.
OFF = "off"
BUDGET = "budget"
ADAPTIVE = "adaptive"
EFFORT = "effort"
TEMPLATE = "template"


class Off(Struct, tag=OFF, tag_field="shape", frozen=True, kw_only=True):
    """Nothing is written: the endpoint is left to whatever it does on its own."""


class Budget(Struct, tag=BUDGET, tag_field="shape", frozen=True, kw_only=True):
    """A number of tokens the model may spend thinking before it answers."""

    tokens: int


class Adaptive(Struct, tag=ADAPTIVE, tag_field="shape", frozen=True, kw_only=True):
    """The model decides how much to think; the request says only that it may."""


class Effort(Struct, tag=EFFORT, tag_field="shape", frozen=True, kw_only=True):
    """One word from :class:`Level`, which is how the endpoints of that family ask for thinking."""

    level: Level


class Template(Struct, tag=TEMPLATE, tag_field="shape", frozen=True, kw_only=True):
    """The chat template's own switch (``enable_thinking``), as a self-hosted server reads it.

    A switch and not a number: what the template does with it is the template's, and this build has no
    business guessing a budget for somebody else's chat template.
    """

    enabled: bool = True


#: Every shape a route may name, in one closed set — the union is what a configuration file is read into.
type ThinkingSpec = Off | Budget | Adaptive | Effort | Template

#: Each shape and its word, in the order a sentence lists them: what a dialect declares it can spell is a
#: set, and a set has no order of its own to put in a message.
WORDS: dict[type, str] = {Off: OFF, Budget: BUDGET, Adaptive: ADAPTIVE, Effort: EFFORT, Template: TEMPLATE}


def said(shapes: frozenset[type]) -> str:
    """What *shapes* are called, for the one sentence that has to refuse a shape a dialect lacks."""
    return ", ".join(word for shape, word in WORDS.items() if shape in shapes)
