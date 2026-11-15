"""One dialect, as two pure functions: a request in, a body out; a body in, a reply out.

A codec is the whole of what is provider-specific. It writes the JSON of one endpoint from the
canonical values (:mod:`plyngent.kernel.model`) and it reads the JSON of one answer back into them, and
it does nothing else — no socket, no clock, no retry, no key. That is what makes a dialect a file and a
row: adding a provider is this module's protocol implemented once, and the loop that asks a model never
learns that a second one exists.

**Pure, so it can be snapshotted.** Every function here takes values and answers values. The body a
request becomes can therefore be compared byte for byte in a test without a server, which is exactly
the comparison a cached prefix rests on (§10.5) and the one a dialect change most needs.

**Errors are classified here, because their shape is the dialect's.** ``{"error": {"message": …}}`` and
``{"type": "error", "error": {…}}`` are two spellings of one thing, and a body that says nothing
readable is still a status. So the status→:class:`~plyngent.kernel.failure.Failure` table lives here,
once, and no dialect invents its own — while *whether to try again* is not decided here at all: that is
:data:`plyngent.kernel.toolresult.OUTCOMES`'s answer for the kind, and the call policy's to act on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.model import ModelFailed

if TYPE_CHECKING:
    from typing import Any

    from plyngent.kernel.lm.caps import Caps
    from plyngent.kernel.model import Reply, Request

#: One request body: the JSON object a dialect writes, before anybody serializes it.
type Body = dict[str, object]

#: What an HTTP status means in the closed set, for the statuses that mean something specific.
#:
#: ``409`` and ``422`` are ``ARGS`` because they say the request was wrong, which for a model call is
#: this kernel's own mistake rather than the model's; everything in the 500s is ``BUSY`` because a
#: server that is unwell is the one thing trying again can fix (§10.11's one table).
_STATUS: dict[int, Failure] = {
    400: Failure.ARGS,
    401: Failure.PERMISSION,
    403: Failure.PERMISSION,
    404: Failure.MISSING,
    408: Failure.TIMEOUT,
    409: Failure.ARGS,
    413: Failure.LIMIT,
    422: Failure.ARGS,
    429: Failure.BUSY,
}


class Codec(ABC):
    """One dialect of the wire, as the two functions that translate a conversation to it and back."""

    #: The row this dialect is filed under, and the word a route names it by.
    name: str = ""
    #: The path this dialect's endpoint lives at, under a route's base — the last thing about the wire
    #: that only a dialect knows. It is a class fact and not a route's, because a route says *which
    #: service* and this says *what any such service answers on*.
    endpoint: str = ""
    #: Whether this dialect has a channel for a picture at all. ``messages`` and ``responses`` do;
    #: ``chat`` does not — a tool result there is text, and the only image channel is a person's message,
    #: which is not who a tool's answer came from. A route whose dialect says ``False`` cannot claim
    #: ``Caps.vision``, and asking one to is answered as blind rather than as a silently dropped picture.
    pictures: bool = False

    @abstractmethod
    def encode(self, request: Request, *, model: str, caps: Caps) -> Body:
        """The request body this dialect wants, for this conversation and this model.

        ``caps`` is what the endpoint can do, and it is the only thing besides the conversation that
        shapes the body: which cache mechanism to carry, how much room to ask for, whether one answer
        may ask for several calls, whether a thinking setting is understood.
        """

    @abstractmethod
    def decode(self, payload: bytes) -> Reply:
        """One answer, read into what a turn acts on.

        A payload this dialect cannot read at all is a :class:`~plyngent.kernel.model.ModelFailed` of
        kind ``PROTOCOL``: the model answered something, and it was not an answer this build speaks.
        """

    @abstractmethod
    def explain(self, payload: bytes) -> str:
        """The sentence an error body carries, or the best one there is when it carries none.

        Per dialect because an error's *shape* is the dialect's. What the status says about the kind of
        trouble is not, so it is classified once by :func:`of_status` and never here.
        """


#: The first status in the range that means "the server is unwell", which is the one thing trying again
#: can fix (§10.11's one table).
_SERVER_ERRORS = 500

#: The first status that is not an answer at all: at or above it, the endpoint refused rather than replied,
#: and what it said is the reason.
FAILURE_STATUS = 400


def of_status(status: int) -> Failure:
    """What kind of trouble an HTTP status is, in the one closed set the kernel has."""
    found = _STATUS.get(status)
    if found is not None:
        return found
    return Failure.BUSY if status >= _SERVER_ERRORS else Failure.PROTOCOL


def refused(codec: Codec, payload: bytes, *, status: int, where: str) -> ModelFailed:
    """A refused request, as the value the turn reports: what kind, and what the endpoint said."""
    said = codec.explain(payload).strip()
    return ModelFailed(of_status(status), f"{where} answered {status}" + (f": {said}" if said else ""))


def decoded(payload: bytes, *, where: str) -> Any:
    """One JSON body, or a ``PROTOCOL`` failure naming the place that could not be read.

    ``Any`` rather than a typed shape because every dialect walks its own document: the one thing they
    share is that a body which is not JSON is not an answer, and that is all this answers. The four
    readers below are what every walk is then made of, and they exist so that no dialect writes an
    ``isinstance`` — a body is somebody else's document, and reading one should not need a type guard
    per field.
    """
    try:
        return msgspec.json.decode(payload)
    except msgspec.DecodeError as exc:
        raise ModelFailed(Failure.PROTOCOL, f"{where} did not answer JSON: {exc}") from exc


def as_dict(value: Any) -> dict[str, Any]:
    """*value* as a JSON object, and the empty one when it is anything else.

    Empty rather than raising, because "this part of the document is missing" is not a failure of the
    whole answer: a provider that omits ``usage`` still answered, and a dialect that raised over it would
    lose the text it did say.
    """
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    """*value* as a JSON array — the same list when it is one, since a caller may extend it."""
    return cast("list[Any]", value) if isinstance(value, list) else []


def as_text(value: Any) -> str:
    """*value* as a string, and nothing at all when it is anything else."""
    return value if isinstance(value, str) else ""


def as_int(value: Any) -> int:
    """*value* as a count, and zero where the document said nothing."""
    return value if isinstance(value, int) else 0


def spelled(arguments: str) -> Any:
    """A call's arguments as the object a dialect wants, falling back to the text that was given.

    A dialect that takes an object (``messages`` and ``responses`` do) needs the arguments parsed; one
    that takes text (``chat``) does not. Parsing here rather than in every dialect keeps the rule that
    *this* layer never judges a model's arguments: text that is not JSON is handed back as text, and the
    tool that was named is the one that reports it as an argument error (see ``kernel.model``).

    No arguments at all is the empty object rather than the empty string: a tool that takes none is
    still called with a package, and a dialect that needs an object there would otherwise be handed text
    that is not one.
    """
    if not arguments.strip():
        return {}
    try:
        return msgspec.json.decode(arguments.encode("utf-8"))
    except msgspec.DecodeError:
        return arguments


def schema(declaration: str) -> Any:
    """One tool's declaration, as the object a dialect writes into a request.

    ``ToolDecl.schema`` is JSON text, and deliberately: text is what can be compared byte for byte, and
    the never-changing tool block is what a cached prefix is anchored on (§10.5). A dialect that needs
    the object parses it here, so there is one parse and one place it can be wrong.
    """
    return msgspec.json.decode(declaration.encode("utf-8"))
