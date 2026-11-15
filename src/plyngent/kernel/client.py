"""The tools a client brings with it: what it offered, and the one way to run one.

A frontend is not only the end of an event stream — it can *do* things, and the things it can do are
its own. So it offers them: a name, the schema of their arguments, and whether they are always there.
The kernel then serves them beside its own tools, in a group of their own that exists exactly as long
as the connection that offered them (§10.8).

Three rules shape this file, and each of them is one sentence from a ruling:

* **A name the kernel answers people through is the kernel's.** ``ask_user_line`` and
  ``ask_user_choice`` are how a question reaches a person, so their *shape* is written here and a
  client may not say what it is. Offering one of those names is how a terminal says "I can ask"; the
  answering itself is the question lane, which already exists.
* **Everybody else's schema is taken as it is.** The kernel adds no field and tightens no constraint,
  because a provider's schema is its promise about its own tool, and the request has to agree with
  what that provider validates against. The cost is real and paid here: a call to such a tool can
  never be *required* to say why, since the provider's schema has no field to say it in (§10.2).
* **A call leaves this process.** So it is approved every time (§10.10), the effect it declares is
  recorded before anyone can tell whether it happened, and what comes back is prose: no status, no
  path and no next step is read out of a client's words, however they are spelled.

What is deliberately *not* here: a way for a client to name itself. The id in ``client:<id>`` is the
host's to give, because a name a party chooses for itself is not an identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Protocol, final

import msgspec
from composat import Component

from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import Effects, Text, ToolResult
from plyngent.kernel.toolspec import ExternalArgs, Intention, Mutation, Res, ToolArgs, ToolSpec
from plyngent.wire.ask import AskSite, Invocation, Returned

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping
    from typing import Any

    from plyngent.kernel.env import Env
    from plyngent.wire.command import OfferedTool
    from plyngent.wire.link import KernelSide

#: The group a client's own tools travel in, named ``client:<id>`` — an offer belongs to one
#: connection, so what it offers is one group that arrives and leaves with it (§10.8).
CLIENT = "client"

#: The names the kernel answers a person through. All three of them have a shape, because the set is
#: about *whose* the name is — a client offers one of these by name and never by schema — and what a
#: question looks like is the kernel's answer, so that a form put to a person reads the same whichever
#: frontend puts it (§10.8, E48/E95).
ASKING_NAMES: frozenset[str] = frozenset({"ask_user_line", "ask_user_choice", "ask_user_form"})


class AskLine(ToolArgs, kw_only=True):
    """Ask the person at the terminal one line of free text, and read their answer back.

    ``question`` is what to put in front of them, and it is the whole prompt: there is no title, no
    field name and no default value that is not visible. ``default`` is suggested and is not the answer
    — an empty line takes it, and with no default an empty line is not an answer at all, so the client
    asks again.
    """

    question: Annotated[str, msgspec.Meta(min_length=1, max_length=1000)]
    default: str = ""


class AskChoice(ToolArgs, kw_only=True):
    """Ask the person at the terminal to pick one of a few options, and read back which one.

    ``options`` is the whole menu: an answer that is not in it is not an answer, and the client asks
    again rather than taking what was typed. ``default`` is a suggested *line*, offered as a
    pre-filled answer, and it is not an extra option.
    """

    question: Annotated[str, msgspec.Meta(min_length=1, max_length=1000)]
    options: Annotated[list[str], msgspec.Meta(min_length=2, max_length=32)]
    default: str = ""


class AskField(msgspec.Struct, frozen=True, kw_only=True, forbid_unknown_fields=True):
    """One line of a form: what the answer is called, what to ask, and what it may be.

    ``options`` non-empty makes it a menu, and then the answer has to be one of them — the same rule
    :class:`AskChoice` puts to a person. ``allow_custom`` beside a menu is the one thing a menu cannot
    say by itself, so such a field is put as a line with the menu written into it and the person's own
    words are taken as they are; no options is a line either way.

    ``default`` is suggested and is not the answer: an empty line takes it, and with no default an empty
    line is not an answer at all, so the client asks again.
    """

    name: Annotated[str, msgspec.Meta(min_length=1, max_length=64)]
    prompt: Annotated[str, msgspec.Meta(min_length=1, max_length=1000)]
    default: str = ""
    options: Annotated[list[str], msgspec.Meta(max_length=32)] = msgspec.field(default_factory=list)
    allow_custom: bool = False


class AskForm(ToolArgs, kw_only=True):
    """Put several things to the person at once, and read back one answer per field.

    A form is a question like the other two rather than a new kind of thing: the person is asked one
    field at a time and what the model reads back is one line per field, named after the field. It
    exists because a model collecting three related answers should not have to spend three calls saying
    they belong together — and because the fields are an array, a frontend that can draw a block of
    labelled fields has something to draw, rather than three names this file happened to pick.

    The form is one answer or none: a person who backs out of the third field has answered nothing, and
    half a form is not something the kernel could name a field of.
    """

    title: Annotated[str, msgspec.Meta(min_length=1, max_length=1000)]
    fields: Annotated[list[AskField], msgspec.Meta(min_length=1, max_length=16)]


#: The asking names and the shape the kernel puts to a person for each of them.
SHAPES: dict[str, type[ToolArgs]] = {
    "ask_user_line": AskLine,
    "ask_user_choice": AskChoice,
    "ask_user_form": AskForm,
}

#: Why an offer of one tool was not taken. Each is the reason an ``OfferRefused`` carries, so a client
#: can say which of the rules it broke instead of guessing from a silent refusal.
DUPLICATE = "a tool of this name already exists, and a name is how the model asks for one"
NAMELESS = "a tool without a name is one nothing could call"
NOT_YOURS = "the kernel has a shape for this name, and a client may not say what it is"
#: Unreachable while every asking name has a shape (a test says so), and kept because the branch below
#: indexes :data:`SHAPES`: a name added to one set and not the other has to be a refusal, not a crash.
NO_SHAPE = "this kernel has no shape for that name yet, so there is nothing to offer"
NOT_AN_OBJECT = "the arguments of a client's tool are a JSON object, and this schema is not one"

#: Why a whole offer was refused, said once for a connection that cannot take one at all.
NO_CAPABILITY = "this connection does not take offers from its client"

#: How much of a call's arguments one declaration carries: a journal line, not a copy of the call.
DECLARED_ARGUMENT_CHARS = 60


class Client(Protocol):
    """The client that offered a tool, as a tool's handler reaches it.

    One verb: run this. The shape of "which client" is the value's business
    (:class:`ClientOffers`), and whether that call is approved, declared and rendered is the
    executor's — so a handler that calls this can be written once for every offered tool.
    """

    async def run(self, tool: str, arguments: str, site: AskSite) -> ToolResult:
        """Run one of this client's tools and answer what came of it.

        ``arguments`` is JSON text, and it is the model's own: the client validates it against the
        schema it declared, because the schema it declared is the only one it promised anything about.
        A call the client never answered is a result and not an exception — the turn goes on, and the
        history says the call did not come back (§10.8, E47).
        """
        ...


@dataclass(frozen=True, slots=True)
class ClientOffers:
    """What one connection offered: the tools, which of them are resident, and the way back to it."""

    id: str
    tools: tuple[ToolSpec, ...]
    resident: frozenset[str]
    #: The connection the tools run over. A live object, like the human channel's attached client, for
    #: the same reason: this value *is* this connection's share of the host.
    link: KernelSide

    @property
    def group(self) -> str:
        """The disclosure group these tools travel in: ``client:<id>``, one per connection (§10.8)."""
        return f"{CLIENT}:{self.id}"

    async def run(self, tool: str, arguments: str, site: AskSite) -> ToolResult:
        """Send one call over the connection and read the answer as a result."""
        returned = await self.link.invoke(
            Invocation(session=site.session, turn=site.turn, tool=tool, arguments=arguments)
        )
        return answered(self.id, tool, arguments, returned)


@final
class Gone:
    """Nobody offers anything: a call that reaches here lost its client before it was sent.

    A value rather than ``None`` for the same reason :class:`~plyngent.kernel.events.NoSink` is one: a
    handler that asked for something it declared should get an answer, not an exception. Nothing was
    sent, so nothing is declared — the external effect recorded on the way out belongs to
    :meth:`ClientOffers.run`, which is the only path that has one.
    """

    async def run(self, tool: str, arguments: str, site: AskSite) -> ToolResult:
        """Answer as a call whose client is not there any more."""
        _ = (arguments, site)
        return ToolResult.of(Failure.CLIENT, f"no client is attached, so {tool} cannot run")


#: The one value to hand a handler when no client is attached.
NO_CLIENT: Client = Gone()


class Offers(Component):
    """What the connection serving this host offers, as a component so attaching it is one ``replace``.

    A component for the same reason the human channel is one: attaching and detaching are the same
    move, a reader never sees a half-attached state, and two hosts cannot answer for each other's
    clients. Nothing attached is a value (:data:`NO_CLIENT`), not a missing field.

    ``issued`` is how many client ids this host has handed out. It lives here rather than in a field
    of its own because the id belongs to the offer: the connection does not name itself, and a name
    that came from the party being named would not be an identity.
    """

    attached: ClientOffers | None
    issued: int

    def __init__(self, attached: ClientOffers | None = None, issued: int = 0) -> None:
        super().__init__()
        self.attached = attached
        self.issued = issued


def specs_from(
    client: str, tools: Iterable[OfferedTool], *, taken: Collection[str]
) -> tuple[tuple[ToolSpec, ...], tuple[tuple[str, str], ...]]:
    """What one offer becomes: the tools it may attach, and the ones refused with why.

    ``taken`` is every name this kernel already has — its own tools, and whatever an earlier offer left
    attached. A name is how a model asks for a tool, so two tools answering to one is refused here
    rather than making which one runs depend on an order.

    The two kinds of tool are decided by *name* and nothing else: a name the kernel answers people
    through carries the kernel's shape, and every other name carries the schema its provider sent.
    """
    group = f"{CLIENT}:{client}"
    attached: list[ToolSpec] = []
    refused: list[tuple[str, str]] = []
    seen: set[str] = set(taken)
    for tool in tools:
        reason = _refusal(tool, taken=seen)
        if reason is not None:
            refused.append((tool.name, reason))
            continue
        seen.add(tool.name)
        attached.append(_asked(tool, group) if tool.name in ASKING_NAMES else _external(tool, group))
    return tuple(attached), tuple(refused)


def declaration(client: str, tool: str, arguments: str) -> str:
    """What left this process, said in one line: ``client c1: run search({"q": …})``.

    It is recorded on the result rather than in a log of its own because it is part of what the call
    did: an effect nobody here can verify, which is exactly what ``Effects.external`` is for.
    """
    return f"client {client}: run {tool}({_short(arguments)})"


def declared(spec: ToolSpec, arguments: str) -> str:
    """The declaration of one call to a client's tool, named after the group the tool travels in.

    The group is ``client:<id>``, so the id is read off the spec rather than off the host: a call is
    declared the same way whether or not that client is still attached, and the executor is not the
    layer that should be asking who is connected.
    """
    return declaration(str(spec.group).rpartition(":")[2], spec.name, arguments)


def answered(client: str, tool: str, arguments: str, returned: Returned) -> ToolResult:
    """One client's answer, read as a result — prose in, prose out, and one declaration.

    The status is the kernel's and the summary is the kernel's: ``ok`` is ``ok``, and everything else is
    :attr:`~plyngent.kernel.failure.Failure.CLIENT`, because an unanswered call and a call the client
    says failed are the same fact here — something was asked of a process this one cannot see into. The
    client's own words go in the body, where they are read as text and never as a status, a path or a
    next step.
    """
    body = Text(value=returned.text) if returned.text else ()
    effects = Effects(external=(declaration(client, tool, arguments),))
    if returned.ok:
        return ToolResult.ok(f"the client ran {tool}", body=body, effects=effects)
    if returned.text:
        return ToolResult.of(Failure.CLIENT, f"{tool} failed on the client", body=body, effects=effects)
    lost = f"{tool} was never answered: the client is gone, or out of time"
    return ToolResult.of(Failure.CLIENT, lost, effects=effects)


def _refusal(tool: OfferedTool, *, taken: Collection[str]) -> str | None:
    """Why this offer of one tool cannot be taken, or ``None`` when it can."""
    if not tool.name:
        return NAMELESS
    if tool.name in taken:
        return DUPLICATE
    if tool.name in ASKING_NAMES:
        if tool.schema:
            return NOT_YOURS
        return None if tool.name in SHAPES else NO_SHAPE
    return None if _is_object(tool.schema) else NOT_AN_OBJECT


def _asked(tool: OfferedTool, group: str) -> ToolSpec:
    """An asking tool: the kernel's shape and the kernel's posture, run over the connection.

    ``Res.CLIENT`` beside ``Res.HUMAN`` says two things at once and both are true: the question is the
    kernel's *and* the answer happens in somebody else's process. The posture is the human rule's —
    ``AUTO`` — because this tool *is* the question, and asking "may I ask?" first would double every
    question in the conversation (§10.10).
    """
    return ToolSpec(
        name=tool.name,
        group=group,
        args=SHAPES[tool.name],
        run=_runs(tool.name),
        uses=Res.CLIENT | Res.HUMAN,
        mutates=Mutation.NONE,
    )


def _external(tool: OfferedTool, group: str) -> ToolSpec:
    """Somebody else's tool: their schema, in their words, and a call that leaves this kernel.

    ``intention`` is spelled out as ``OPTIONAL`` rather than derived, and that is the honest price of
    the rule above it: ``Mutates.OUTSIDE`` would derive ``REQUIRED``, and a required field the provider
    never declared would make every call of the tool fail to decode (§10.2, §10.8 rule 6).
    """
    return ToolSpec(
        name=tool.name,
        group=group,
        args=ExternalArgs(schema=_schema(tool.schema)),
        run=_runs(tool.name),
        uses=Res.CLIENT,
        mutates=Mutation.OUTSIDE,
        intention=Intention.OPTIONAL,
    )


def _runs(name: str) -> Callable[[Env, Any], Awaitable[ToolResult]]:
    """The one handler an offered tool runs: hand the call to the client that offered it.

    One function for every offered tool, closed over its name, because nothing in the kernel knows what
    any of them does — and a per-tool handler would be the kernel pretending it did. It reaches the
    client through :attr:`~plyngent.kernel.env.Env.client`, so the declaration is checked like any
    other resource.
    """

    async def run(env: Env, args: Any) -> ToolResult:
        return await env.client.run(name, _arguments(args), env.site("client"))

    return run


def _arguments(args: Any) -> str:
    """The decoded package as the JSON text the client declared a schema for."""
    return msgspec.json.encode(args).decode()


def _schema(text: str) -> Mapping[str, Any]:
    """A client's JSON Schema as the object it already was, read once so it is read consistently."""
    return json.loads(text)


def _is_object(text: str) -> bool:
    """Whether this is the text of a JSON object — the one thing asked of somebody else's schema."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return False
    return isinstance(parsed, dict)


def _short(arguments: str) -> str:
    """A call's arguments as much as fits in one line of a journal, with the rest elided."""
    flat = " ".join(arguments.split())
    return flat if len(flat) <= DECLARED_ARGUMENT_CHARS else f"{flat[:DECLARED_ARGUMENT_CHARS]}…"
