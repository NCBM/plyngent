"""How big a request is, before it is sent, as a number a turn can act on.

A budget needs a size, and the only exact size is the provider's own count — which arrives *after* the
request it describes and cannot be asked about one that has not been sent. So this is an estimate, and it
is honest about being one: characters divided by a rule of thumb, with the one thing that is not text
counted as a floor. What makes it usable is that it is **paired**: a request's estimate is written beside
the ``prompt_tokens`` the provider charged for that same payload
(:attr:`plyngent.kernel.trail.Usage.estimate`), so the error between the two is a measurement rather than a
guess, and a caller that needs tokens rather than characters scales by it.

Three decisions, none of them a detail:

* **Characters, not tokens.** There is no tokenizer here and there cannot be one: every provider has its
  own, and a build that guessed one would be wrong about the numbers it exists to be right about while
  looking exact. Characters per token is a constant with an error of a few tens of percent, which the
  pairing above turns into a calibration.
* **A part counts when it travels.** A message's text, the calls it asked for, the thinking a dialect is
  handed back, the tool declarations, the system prompt — what a provider is given. A result's document
  handles are already in its text and are not counted twice.
* **A picture is a floor, not a measurement.** Pixels cost what the far end's own encoder decides, which
  nothing here can know. A fixed number per picture keeps an image from being free in a budget without
  pretending to know what it costs.

What is deliberately not here: the policy. This module answers "how big", and nothing below it decides
what to do when the answer is too big — that is the turn's, where a request is assembled and the room a
route gives it is known.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.model import Assistant, Result

if TYPE_CHECKING:
    from collections.abc import Iterable

    from plyngent.kernel.model import AnyMessage, ToolDecl

#: How many characters go to a token, on average, across the languages and the text this program sends.
#: The value every rough count starts from; it is a rule of thumb and is treated as one.
CHARS_PER_TOKEN = 4

#: What one picture is counted as. Not a measurement — a floor, so an image is never free in a budget.
PICTURE_TOKENS = 1_024


def text_chars(message: AnyMessage) -> int:
    """How many characters of *message* a provider is handed."""
    total = len(message.text)
    if isinstance(message, Assistant):
        for call in message.calls:
            total += len(call.id) + len(call.tool) + len(call.arguments)
        for thought in message.thoughts:
            total += len(thought.text) + len(thought.signature) + len(thought.sealed)
    elif isinstance(message, Result):
        total += len(message.call) + len(message.tool)
    return total


def pictures(message: AnyMessage) -> int:
    """How many pictures *message* carries. Only a result can, and only for the route that reads them."""
    return len(message.pictures) if isinstance(message, Result) else 0


def estimate(
    messages: Iterable[AnyMessage],
    *,
    system: str = "",
    tools: Iterable[ToolDecl] = (),
) -> int:
    """How many tokens this request is likely to be counted as, rounded up.

    ``system`` and ``tools`` are what the request carries besides the conversation, and they are here
    rather than added by the caller because they travel to the same place and are counted by the same
    number at the other end: a tool block of a thousand characters is two hundred and fifty tokens of
    ``prompt_tokens`` whoever counted them.
    """
    chars = len(system)
    for declared in tools:
        chars += len(declared.name) + len(declared.schema)
    images = 0
    for message in messages:
        chars += text_chars(message)
        images += pictures(message)
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN + images * PICTURE_TOKENS
