"""A model a story can script, and the record of what it was asked.

A story's model has to be two things at once: the thing the kernel asks, and the thing the story reads
back. So the replies are *scheduled* rather than computed — the story says what the model answers and in
what order, and a request past the end of the script is a failure rather than an invented answer — apart
from one kind of step, which is chosen by what the model was just read (:data:`Step`). That kind exists
because a tool's answer can carry a value the story cannot know before the call runs, and the call after
it has to take that value the way a model would: out of the request, not out of the test's bookkeeping.
Every request is kept, because "what the model saw" is one of the four faces a story tests (§11).

``Held`` is the one kind of reply that does not answer at once. It exists so a story can have a request
in flight while it does something else — press Ctrl+C, send a second message — which is otherwise a race
no test can win.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from plyngent.kernel.model import Call, Model, Reply

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from plyngent.kernel.model import Request


def text(said: str) -> Reply:
    """A reply that only says something."""
    return Reply(text=said)


def calling(tool: str, arguments: dict[str, object] | str, *, says: str = "", call: str = "") -> Reply:
    """A reply that says ``says`` and asks for one call of ``tool``.

    ``arguments`` is the tool's own package, as the model would spell it: a dict is encoded here, and a
    string is taken as already-written JSON (for a story about arguments that do not decode).
    """
    spelled = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return Reply(text=says, calls=(Call(id=call or tool, tool=tool, arguments=spelled),))


@dataclass(frozen=True, slots=True)
class Held:
    """A reply that waits until the story lets it go, so a request can be in flight while the story acts.

    ``ready`` belongs to the story: it sets it when it wants the answer to arrive. A cancelled request
    never gets there, which is exactly what a story about interruption is asking for.
    """

    reply: Reply
    ready: asyncio.Event = field(default_factory=asyncio.Event)


#: One step of a script: a reply, a reply that waits, or a reply chosen by what the model was told.
#:
#: The third kind is there for the values a story cannot write down in advance. A call can hand back a
#: name or a handle that the next call has to carry — a terminal's session, a document to continue — so
#: the next reply is decided from the request the moment it arrives, which is the only place that value
#: exists. The step answers a reply, so a decided reply can still be a :class:`Held` one.
type Step = Reply | Held | Callable[[Request], Reply | Held]


def told(request: Request) -> str:
    """What the model was read last: the newest message's text, or nothing where there is no history.

    Every kind of message has a text — a person's line, the model's own answer, a tool's result — so this
    is the one reader that does not care which kind it was handed. It is also how a decided step reads a
    value the call before it produced, since that value is in the text the model was given and nowhere
    else.
    """
    return request.messages[-1].text if request.messages else ""


class ScriptedModel(Model):
    """The steps a story scheduled, and every request that asked for one."""

    _script: list[Step]
    _asked: list[Request]

    def __init__(self, script: Iterable[Step]) -> None:
        self._script = list(script)
        self._asked = []

    @property
    def asked(self) -> tuple[Request, ...]:
        """Every request, in the order the kernel made them."""
        return tuple(self._asked)

    @override
    async def respond(self, request: Request) -> Reply:
        """Answer the next step of the script; a request the story did not script is a failed story."""
        self._asked.append(request)
        if not self._script:
            msg = f"the model was asked {len(self._asked)} times and the story scripted {len(self._asked) - 1} replies"
            raise AssertionError(msg)
        step = self._script.pop(0)
        if callable(step):
            step = step(request)
        if isinstance(step, Held):
            await step.ready.wait()
            step = step.reply
        return step
