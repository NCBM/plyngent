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
no test can win. ``Streamed`` is an answer told in pieces, held between them the same way, so a story can
read the screen while the answer is still being said.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from plyngent.kernel.failure import Failure
from plyngent.kernel.model import Call, Delta, Model, ModelFailed, Reply

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterable

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


@dataclass(frozen=True, slots=True)
class Streamed:
    """A reply that is said in pieces, and — when the story wants to look — stops between them.

    ``pieces`` are the fragments as the model says them and ``reply`` is what they add up to, which is
    what a real streamed answer hands over too. ``ready`` belongs to the story, as it does on
    :class:`Held`: it is what lets a story read the screen *while the answer is still coming*, which is
    the one thing about a stream that is invisible once the turn has ended. ``None`` is a stream that
    runs straight through, which is the ordinary case.
    """

    pieces: tuple[str, ...]
    reply: Reply
    ready: asyncio.Event | None = None


@dataclass(frozen=True, slots=True)
class Failing:
    """A step that does not answer: the model could not be asked, and how.

    A scripted model that only ever answered could not play the one case the route layer exists for —
    an endpoint that is busy, that refuses the key, that answers nonsense — so a story says which kind of
    trouble it wants and the model reports it the way a real one would: as a value the turn can act on,
    not as a traceback.
    """

    failure: Failure
    says: str


#: One step of a script: a reply, a reply that waits, a reply in pieces, a request that cannot be
#: answered, or a reply chosen by what the model was told.
#:
#: The last kind is there for the values a story cannot write down in advance. A call can hand back a
#: name or a handle that the next call has to carry — a terminal's session, a document to continue — so
#: the next reply is decided from the request the moment it arrives, which is the only place that value
#: exists. The step answers a reply, so a decided reply can still be a :class:`Held` one.
type Step = Reply | Held | Streamed | Failing | Callable[[Request], Reply | Held]


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
        step = self._next(request)
        if isinstance(step, Streamed):
            # A streamed step still has an answer in it: what was said, without the pieces it was said in.
            return step.reply
        return await _settled(step)

    @override
    def stream(self, request: Request) -> AsyncIterator[Delta | Reply]:
        """Answer the next step, in the pieces it is said in — one piece for a step that is not streamed."""
        return self._streamed(self._next(request))

    async def _streamed(self, step: Reply | Held | Streamed | Failing) -> AsyncIterator[Delta | Reply]:
        """The pieces of one step, and then the answer they add up to."""
        if not isinstance(step, Streamed):
            yield await _settled(step)
            return
        for piece in step.pieces:
            yield Delta(text=piece)
            if step.ready is not None:
                await step.ready.wait()
        yield step.reply

    def _next(self, request: Request) -> Reply | Held | Streamed | Failing:
        """The next step of the script, with a decided one read: asking past its end is a failed story."""
        self._asked.append(request)
        if not self._script:
            msg = f"the model was asked {len(self._asked)} times and the story scripted {len(self._asked) - 1} replies"
            raise AssertionError(msg)
        step = self._script.pop(0)
        return step(request) if callable(step) else step


async def _settled(step: Reply | Held | Failing) -> Reply:
    """One step once it has arrived: a held one waits first, and one that failed raises what it said."""
    if isinstance(step, Held):
        await step.ready.wait()
        step = step.reply
    if isinstance(step, Failing):
        raise ModelFailed(step.failure, step.says)
    return step
