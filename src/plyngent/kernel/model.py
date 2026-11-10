"""What a conversation is made of, and what asking a model for the next turn of one looks like.

A message is a value, not a view. The journal projects its own entries into these
(:meth:`plyngent.kernel.trail.Trail.derive`), a request hands a tuple of them over, and a scripted
model is read the same tuple a provider will one day encode. Nothing here knows a provider, a
transport or a JSON dialect: the canonical encoding is R11's business, and it stays out of this
module because the history has to be spellable without one.

Three shapes and no more, because they are the three a model API has ever distinguished: a person's
line, the model's own answer — which may carry the calls it asked for — and what one call came back
with.

Two rules the shapes hold to:

* **A call's arguments stay text.** The model produced text; decoding it into a tool's package is
  that tool's own step (:meth:`plyngent.kernel.toolspec.ToolSpec.decode`), so a malformed call is an
  argument error of the tool that was named rather than a failure of the request that carried it.
* **A call is named by the model, not by the tool.** ``Call.id`` is what a result quotes back, and it
  has to be, because one answer may ask for the same tool twice and two results cannot be told apart
  by the tool's name.

The asking itself is three values and one component: :class:`Request` is what goes out,
:class:`Reply` what comes back, :class:`Model` the two-verb contract, and
:class:`ModelChannel` the host's slot for whichever model is installed. What is *not* here is as
deliberate as what is: there is no provider, no route, no retry and no token accounting. Those are
R11's and R13's, and a turn that had to know them would be a turn that could not be run against a
script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from composat import Component
from msgspec import Struct

from plyngent.kernel.toolresult import Status


class Call(Struct, frozen=True, kw_only=True):
    """One tool call the model asked for, in the three things anything downstream needs.

    ``id`` is opaque and comes from whoever made the answer: a provider supplies one, a scripted model
    is given one. Nothing here invents an order or a meaning for it — it is a correlation handle, and
    :class:`Result` is what closes the pair.
    """

    id: str = ""
    tool: str = ""
    #: The arguments as the model spelled them: JSON text, not yet a package.
    arguments: str = ""


class Message(Struct, frozen=True, kw_only=True, tag_field="role"):
    """One thing in the conversation.

    Never instantiated as itself: the three kinds below are what a history holds, and the tag is what
    lets one tuple carry all three — a projection, a request and a journal row are then the same kind
    of value, and nothing has to switch on a field called ``kind`` and hope.
    """


class User(Message, tag="user", frozen=True, kw_only=True):
    """A line a person said, as the model reads it back."""

    text: str = ""


class Assistant(Message, tag="assistant", frozen=True, kw_only=True):
    """What the model answered: something it said, and the calls it asked for, in one message.

    One message rather than two — a text and a set of calls — because they arrived together: a request
    that named three tools is one answer, and splitting it would let an interruption leave calls
    attributed to a message that never arrived.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()


class Result(Message, tag="tool-result", frozen=True, kw_only=True):
    """What one tool call came back with, in the words the model reads.

    ``text`` is the rendered body, not the parts it was rendered from: a projection turns parts into
    one text (:mod:`plyngent.kernel.render`), and this is the value after that step. ``status`` is
    kept because it is the one thing the text cannot be trusted to start with, and the person's
    timeline needs it whether or not the model does.
    """

    call: str = ""
    tool: str = ""
    status: Status = Status.OK
    text: str = ""


#: What a history is made of, as one type a tuple can be — for a reader that switches on the kind.
type AnyMessage = User | Assistant | Result

#: The tags a model reads, which is what a projection keeps. A stored entry carrying anything else is
#: bookkeeping — a prune, a state change — and the projector skips it unread, so the vocabulary a log may
#: hold is wider than the one a model is sent.
MODEL_TAGS: frozenset[str] = frozenset({"user", "assistant", "tool-result"})


class ToolDecl(Struct, frozen=True, kw_only=True):
    """One tool as a model is told about it: what to call it, and the schema of its arguments.

    ``schema`` is JSON text rather than a structure because that is what a provider's request carries,
    and because text is what can be compared byte for byte: the tool block sits in front of everything
    else in a prompt, so "the same declarations produce the same bytes" is the property a cached prefix
    rests on (§10.5). The description travels inside the schema — ``msgspec`` writes the argument
    package's docstring into ``description`` — so there is one declaration and not two.
    """

    name: str = ""
    schema: str = ""


class Request(Struct, frozen=True, kw_only=True):
    """What a turn asks a model for: the conversation as a whole, and the tools it may reach for.

    ``messages`` is the whole history the model is given, not a delta. There is no incremental protocol
    here, and "what the model saw" has to be reproducible from the journal alone — a request carrying the
    delta would make the answer depend on a second list that had to be kept in step.
    """

    messages: tuple[AnyMessage, ...] = ()
    tools: tuple[ToolDecl, ...] = ()


class Reply(Struct, frozen=True, kw_only=True):
    """What one model answered: something it said, and the calls it asked for.

    Both in one value rather than a union of two kinds, because a model may do both at once — say a line
    and ask for a tool in the same answer — and a union would force this layer to invent a precedence.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()


class NoModel(Exception):  # noqa: N818 — a missing model is a state of the kernel, not a failure of a call
    """There is nothing to ask: a kernel brought up without a model refuses turns rather than hanging."""


class Model(ABC):
    """What a turn asks: one request in, one reply out.

    Nothing about a provider here — no dialect, no transport, no streaming. That is R11's canonical
    encoding, and keeping it out means a scripted model in a test and a real one in a run are asked the
    same question in the same words.
    """

    @abstractmethod
    async def respond(self, request: Request) -> Reply:
        """Answer one request."""


class ModelChannel(Component, Model):
    """The model a host asks, as a component so that installing one is a declaration and swapping it is a
    ``replace`` — the same move that attaches a client to :class:`~plyngent.kernel.events.HumanChannel`.

    ``None`` is "nothing is installed", and it is a value rather than a field nobody set, so the turn can
    refuse before it starts instead of failing at the first request. A kernel that answers turns without a
    model would have to invent what the model said.
    """

    inner: Model | None

    def __init__(self, inner: Model | None = None) -> None:
        super().__init__()
        self.inner = inner

    @property
    def installed(self) -> bool:
        """Whether there is a model to ask."""
        return self.inner is not None

    @override
    async def respond(self, request: Request) -> Reply:
        """Ask whoever is installed; a request that reached here with nobody installed is a bug."""
        inner = self.inner
        if inner is None:
            msg = "no model is installed, so nothing answered"
            raise NoModel(msg)
        return await inner.respond(request)
