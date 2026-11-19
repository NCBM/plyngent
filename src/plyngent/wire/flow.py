"""The one thing a client owes the transport: how far it has rendered.

``Ack`` is not a command and not an event. It goes to the *connection*, not to a session, and the
kernel's link consumes it before anything else sees it — so it never reaches a turn. What it carries is
one mark per tree it has been read in, because the numbers are per tree (``Mark``). Three jobs:

* **resume.** A client that comes back says how far it had got on the connection that died, and the
  kernel answers with what the host said since — once, and only for a number that is a claim about the
  past rather than progress in the present (``resume``). Without that word, every ack would be asking
  to be sent what it is already holding.
* **liveness.** A client that stops acking while the backlog grows is presumed gone; the link then
  drops the connection, refuses that client's questions and puts the human back to "nobody to ask".
  Judging by silence needs a number that arrives regularly, and this is it.
* **coalescing.** Because the client says what it has, the link knows how far behind it is and may
  fold what is queued into fewer, larger messages. Without an ack there is no way to tell "slow" from
  "gone", and both answers would be guesses.

The numbers are the host's, per tree (:class:`~plyngent.kernel.ledger.Ledger`), because a number that
cannot be compared across a reconnect is a number a reconnect cannot use — and because two trees keep
two counts, one connection may be looking at more than one of them, and "how far have I read" then has
one answer per tree. That is why an ack carries a set of marks and not a number (E154).
"""

from __future__ import annotations

from msgspec import Struct


class Mark(Struct, frozen=True, kw_only=True):
    """How far a client has rendered one tree's events: a number, and which count it belongs to.

    ``workspace`` names the tree the number was counted in, empty meaning the connection's default one.
    ``session`` names the conversation it was rendered in, which is what lets a frontend that reopens a
    single conversation resume that one alone: the counts stay per tree, and the session only says which
    of them a client was looking at.
    """

    workspace: str = ""
    session: str = ""
    seq: int = 0


class Ack(Struct, frozen=True, kw_only=True):
    """Everything up to and including each mark has been rendered; nothing before them needs replaying."""

    marks: tuple[Mark, ...] = ()
    #: Whether these marks are a claim about a connection that is gone — the first frame of a client that
    #: came back — rather than how far this one has got. Only the first claim on a connection is answered.
    resume: bool = False
