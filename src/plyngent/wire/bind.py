"""The first thing a client says: which version of the contract it speaks.

A greeting tells a client what the kernel is; this is the other half of the same exchange, and the number
in it is there for the same reason the greeting's is: the shape of every value after it depends on the
two ends agreeing about it. A client that cannot be read is one whose values must not be guessed at, so
it has to be able to say so *before* it says anything else — which is why this is a lane of its own and
not a command. A command is one word of a union the kernel defines, and a client whose version a kernel
does not speak has to be able to say so without having sent one.

It says nothing else. Which tree a connection works in, and which conversation it starts in, are
commands: they are choices a client makes after it has been told what this kernel can do.
"""

from __future__ import annotations

from msgspec import Struct

from plyngent.wire.manifest import PROTOCOL_VERSION


class Bind(Struct, frozen=True, kw_only=True):
    """What this client is, as far as the contract goes.

    The number defaults to this build's own, so "this program, today" needs no argument — the same move
    :class:`~plyngent.wire.manifest.KernelManifest` makes with its copy of it. A kernel that is sent a
    version it does not speak does not serve the connection, and a client that is greeted with one it
    does not speak does not send this at all: neither side answers an unknown shape with a guess.
    """

    protocol_version: int = PROTOCOL_VERSION
