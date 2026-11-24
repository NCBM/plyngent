"""Tools a program outside this one answers for, in the three parts a provider needs.

A group of tools can be somebody else's program, and that is one thing said three ways:

* ``spec`` is what a server's listing becomes here — a name this kernel can be asked by, and the
  schema its own tool declares.
* ``result`` is what its answer becomes — prose in somebody else's words, and *whether it worked* as
  the kernel's own kinds of failure.
* ``server`` is the provider: the child process, the handshake, and the call that leaves this one.

Only the third has a lifetime. The first two are pure, which is why the protocol's shape can be tested
without a program on the other end of a pipe.
"""

from __future__ import annotations
