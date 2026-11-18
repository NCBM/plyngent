"""What a client learns at handshake time, once.

The manifest is the kernel describing itself: which commands exist (so a client can build its parser
and its completion instead of keeping a second list), which reads exist (so a frontend with buttons
instead of a line has the same table), what it can do, and the two numbers that make the liveness rule
operative. It is read once per connection and kept; nothing in it changes while a connection lives.

``ack_interval`` and ``dead_after`` are negotiated rather than hard-coded on both sides for the same
reason the protocol version is: a client that cannot be trusted to ack every 2 seconds is a client,
not a bug. ``dead_after`` is therefore the kernel's own budget (``k * ack_interval``, with ``k`` at
least 3), and the kernel alone decides that a silent peer is gone.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast, get_args

import msgspec
from msgspec import Struct

from plyngent.wire.command import AnyCommand
from plyngent.wire.query import AnyQuery

if TYPE_CHECKING:
    from plyngent.wire.command import Command
    from plyngent.wire.query import Query

#: Bumped whenever a wire type changes shape. A client whose major differs is refused, not adapted.
PROTOCOL_VERSION = 1

#: Defaults the kernel offers; a manifest may carry different values.
DEFAULT_ACK_INTERVAL = 2.0
DEFAULT_DEAD_AFTER = 8.0

#: The capability a client looks for before it offers anything: without it the kernel refuses every
#: ``Offer`` it is sent. A kernel that cannot run a client's tools must not be told about them, so the
#: answer is in the document a client reads at handshake time rather than in a refusal after the fact.
CLIENT_OFFER = "client-offer"


class CommandSpec(Struct, frozen=True, kw_only=True):
    """One command, as a client needs it: a tag to send, a line to show, an argument shape.

    ``args`` is the JSON Schema of the command's own arguments, as msgspec generates it for the struct
    the kernel decodes into — one document, ``$defs`` included, so the parser, the help text and the
    validator cannot disagree.
    """

    name: str = ""
    summary: str = ""
    args: str = ""


class QuerySpec(Struct, frozen=True, kw_only=True):
    """One read, as a client needs it: the same three things a :class:`CommandSpec` carries.

    Beside the commands because a frontend with buttons has no line to type a word on: ``/grants`` and a
    button in a window are the same question, and only one of them needs this program's parser (E155).
    What is deliberately absent is which reply comes back: a reply is its own tagged union, so the answer
    says what it is when it arrives rather than in a table that would have to be kept in step with it.
    """

    name: str = ""
    summary: str = ""
    args: str = ""


class KernelManifest(Struct, frozen=True, kw_only=True):
    """The kernel's description of itself."""

    protocol_version: int = PROTOCOL_VERSION
    kernel: str = "plyngent"
    #: Which run of this kernel this is. A client that remembered how far it had read on a *previous*
    #: run is holding a number from another count, so it compares this first and claims nothing when it
    #: differs (§5 B).
    epoch: str = ""
    #: The directory this kernel works in, as it resolved it. A client that was told which workspace to
    #: work in compares rather than assumes: two programs in different directories are two kernels, and
    #: finding that out here is a sentence instead of a session that edits the wrong tree (§5 B, E123).
    workspace: str = ""
    commands: tuple[CommandSpec, ...] = ()
    #: The reads, beside the commands: a person asking "which paths may this reach" is one question
    #: whether it is asked with a slash word or a button (E155).
    queries: tuple[QuerySpec, ...] = ()
    #: What this build can do — "pty", "notify", "client-offer", … Read by the client, never by a turn.
    caps: tuple[str, ...] = ()
    ack_interval: float = DEFAULT_ACK_INTERVAL
    dead_after: float = DEFAULT_DEAD_AFTER


def command_specs() -> tuple[CommandSpec, ...]:
    """The command table, derived from the union a client already decodes into.

    This is why the table is not a second list: the name is the tag msgspec dispatches on, the
    argument schema is the struct's own, and the one-line help is the first line of the class's
    docstring. A command added to the union appears here without anybody remembering to add it, and a
    command whose help nobody wrote fails the test that every spec has a summary.
    """
    return tuple(_spec(cls) for cls in get_args(AnyCommand.__value__))


def query_specs() -> tuple[QuerySpec, ...]:
    """The read table, derived the same way from the union a client sends its questions from.

    The same claim as :func:`command_specs` — the tag, the first line of the docstring, the struct's own
    schema — and one difference that is the query's own: the number on it belongs to the connection, so
    it is dropped from the schema a client is handed. A table that told a frontend to choose one would be
    telling it how to answer somebody else's question (:mod:`plyngent.wire.query`).
    """
    return tuple(_read(cls) for cls in get_args(AnyQuery.__value__))


def _spec(cls: type[Command]) -> CommandSpec:
    """One command as a client needs it: what to send, what it takes, and what it is for."""
    return CommandSpec(
        name=_tagged(cls),
        summary=_summary(cls),
        args=json.dumps(msgspec.json.schema(cls), sort_keys=True, separators=(",", ":")),
    )


def _read(cls: type[Query]) -> QuerySpec:
    """One read as a client needs it: what to send, what it takes, and what it is for."""
    return QuerySpec(
        name=_tagged(cls),
        summary=_summary(cls),
        args=json.dumps(_without_id(msgspec.json.schema(cls)), sort_keys=True, separators=(",", ":")),
    )


def _tagged(cls: type[Command | Query]) -> str:
    """The word this kind of value carries, which is what a client sends in place of a name."""
    tag = cls.__struct_config__.tag
    if not isinstance(tag, str):
        msg = f"{cls.__name__} carries no string tag, so a client has no word for it"
        raise TypeError(msg)
    return tag


def _summary(cls: type[Command | Query]) -> str:
    """The one line a table row can hold: what this word is for, as its author wrote it."""
    doc = (cls.__doc__ or "").strip()
    return doc.splitlines()[0] if doc else ""


def _without_id(schema: dict[str, Any]) -> dict[str, Any]:
    """A read's arguments without the reply's number: that one is stamped by the link, not chosen."""
    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    definitions = cast("dict[str, Any]", schema["$defs"])
    named = cast("dict[str, Any]", definitions[reference.rsplit("/", 1)[-1]])
    properties = cast("dict[str, Any]", named.get("properties") or {})
    _ = properties.pop("id", None)
    return schema
