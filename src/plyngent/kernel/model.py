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
:class:`ModelChannel` the host's slot for whichever model is installed. What a model *costs*
(:class:`Usage`), where it was *cut off* (:attr:`Reply.cut`), which *route* answered
(:attr:`Request.route`) and what the *cache* is keyed on (:attr:`Request.cache_key`) are fields here,
because they are what a turn reports and what a person reads — but how any of them is obtained is not:
a dialect, an HTTP call, a retry and a timeout live in :mod:`plyngent.kernel.lm`, behind the same two
verbs, so a turn that had to know a provider would be a turn that could not be run against a script.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import override

from composat import Component
from msgspec import Struct

# Not a type-only import: ``msgspec`` resolves a struct's field annotations from this module's globals
# when the class is created, so a lazily-undefined name would make ``Reply`` unencodable.
from plyngent.kernel.failure import Failure  # noqa: TC001
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


class Picture(Struct, frozen=True, kw_only=True):
    """An image, as the bytes a provider is handed and the media type those bytes are.

    It is not a part of a result's body, and that is the whole point: a body too big to send is *kept*,
    and the conversation carries its handle (:class:`~plyngent.kernel.toolresult.Document`). Pixels are
    the one content a model can read and a handle cannot stand in for, so they are fetched when a
    request is assembled and travel only in the value that is handed over. ``Result.pictures`` is
    therefore empty for every row the journal holds and filled for the value a model is asked with —
    which is what keeps "a body never enters a row" true while a model still sees an image.
    """

    media: str = ""
    data: bytes = b""


class Result(Message, tag="tool-result", frozen=True, kw_only=True):
    """What one tool call came back with, in the words the model reads.

    ``text`` is the rendered body, not the parts it was rendered from: a projection turns parts into
    one text (:mod:`plyngent.kernel.render`), and this is the value after that step. ``status`` is
    kept because it is the one thing the text cannot be trusted to start with, and the person's
    timeline needs it whether or not the model does.

    ``documents`` is the other half of that body: the handles the parts kept, in the order they came
    back. The text says the same thing in words a model reads, and this says it in a form a request can
    act on — a handle is how a picture reaches the model at all (§10.5, E56). ``pictures`` is what
    those handles were fetched into, and it is the one field here that a stored row never carries.
    """

    call: str = ""
    tool: str = ""
    status: Status = Status.OK
    text: str = ""
    documents: tuple[str, ...] = ()
    pictures: tuple[Picture, ...] = ()


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

    ``system`` is the part of the prompt that is not the conversation: the sections a host was installed
    with — who the model is, the tool playbook, one section per group whose tools the session can reach,
    the skills this program can see — ending with the one line saying which groups this session has not
    been shown (§10.4, §0.10 E63). It is a field rather than a message because nobody said it — a history
    that carried it would hand it back to the model as its own words on the next turn.
    """

    system: str = ""
    messages: tuple[AnyMessage, ...] = ()
    tools: tuple[ToolDecl, ...] = ()
    #: Which route answers: the name a session's setting resolves to, never a provider's own name.
    #: A model reads it to know where to send the request, and everything else — the tool block that
    #: was built, the caps that gated a picture — is already decided by the time it is here.
    route: str = ""
    #: The short stable key a provider's prefix cache is keyed on (§10.5): the session, and the two
    #: counters that say when its prompt moved. Built where the prompt is assembled and not in a codec,
    #: because the numbers are facts about a session and a second place computing them could disagree.
    cache_key: str = ""


class Usage(Struct, frozen=True, kw_only=True):
    """What one request cost, and how much of it a provider could answer from what it already had.

    ``cached`` is the number §10.5 exists for: a prompt whose prefix nobody moved is cheaper, and the
    only way to know it was is for the provider to say so. ``None`` is that provider saying nothing —
    kept apart from ``0`` on purpose, because "nothing was cached" and "nobody told us" are different
    facts and a report that mixed them would be uncountable.
    """

    prompt: int = 0
    completion: int = 0
    cached: int | None = None

    @property
    def total(self) -> int:
        """Both halves, as the one number a bill is written in."""
        return self.prompt + self.completion

    @property
    def reported(self) -> bool:
        """Whether anybody said anything at all, which is not the same as saying zero.

        A route that does not count tokens leaves every field where it started, and a turn that said
        "0+0 tokens" about such a request would be reporting a fact it does not have — and printing it on
        every turn. So nothing is said unless something was.
        """
        return bool(self.prompt or self.completion or self.cached is not None)


class Reply(Struct, frozen=True, kw_only=True):
    """What one model answered: something it said, and the calls it asked for.

    Both in one value rather than a union of two kinds, because a model may do both at once — say a line
    and ask for a tool in the same answer — and a union would force this layer to invent a precedence.

    ``cut`` is why an answer stopped short when it did: a provider's finish reason, in the closed set
    the rest of the kernel already speaks, so "it hit the output limit" and "the provider refused the
    content" reach a person in the same words a failed call would. ``None`` is an answer that ended on
    its own, which is not the same as one that ran out of room mid-sentence.
    """

    text: str = ""
    calls: tuple[Call, ...] = ()
    usage: Usage = Usage()
    cut: Failure | None = None


class NoModel(Exception):  # noqa: N818 — a missing model is a state of the kernel, not a failure of a call
    """There is nothing to ask: a kernel brought up without a model refuses turns rather than hanging."""


class ModelFailed(Exception):  # noqa: N818 — the asking failed, and the turn reports it rather than raising
    """A model could not be asked, and which kind of trouble that was.

    Not a reply: a reply is what a model said, and a model that never answered said nothing. So the two
    are different things, and this one carries the pair a refused tool result carries — the closed-set
    kind, and the one sentence a person reads. What is deliberately absent is a retry decision: whether
    to ask again is the call policy's, and it lives next to the asking (:mod:`plyngent.kernel.lm`), not
    at the raise site.
    """

    failure: Failure
    says: str

    def __init__(self, failure: Failure, says: str) -> None:
        super().__init__(says)
        self.failure = failure
        self.says = says


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
