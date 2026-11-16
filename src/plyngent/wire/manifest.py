"""What a client learns at handshake time, once.

The manifest is the kernel describing itself: which commands exist (so a client can build its parser
and its completion instead of keeping a second list), what it can do, and the two numbers that make
the liveness rule operative. It is read once per connection and kept; nothing in it changes while a
connection lives.

``ack_interval`` and ``dead_after`` are negotiated rather than hard-coded on both sides for the same
reason the protocol version is: a client that cannot be trusted to ack every 2 seconds is a client,
not a bug. ``dead_after`` is therefore the kernel's own budget (``k * ack_interval``, with ``k`` at
least 3), and the kernel alone decides that a silent peer is gone.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, get_args

import msgspec
from msgspec import Struct

from plyngent.wire.command import AnyCommand

if TYPE_CHECKING:
    from plyngent.wire.command import Command

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


def _spec(cls: type[Command]) -> CommandSpec:
    """One command as a client needs it: what to send, what it takes, and what it is for."""
    tag = cls.__struct_config__.tag
    if not isinstance(tag, str):
        msg = f"{cls.__name__} carries no string tag, so a client has no word for it"
        raise TypeError(msg)
    doc = (cls.__doc__ or "").strip()
    return CommandSpec(
        name=tag,
        summary=doc.splitlines()[0] if doc else "",
        args=json.dumps(msgspec.json.schema(cls), sort_keys=True, separators=(",", ":")),
    )
