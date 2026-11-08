"""The one thing a client owes the transport: how far it has rendered.

``Ack`` is not a command and not an event. It goes to the *connection*, not to a session, and the
kernel's link consumes it before anything else sees it — so it carries no session and never reaches a
turn. Three jobs in one number:

* **resume.** ``resume(after=seq)`` replays what was missed from that point.
* **liveness.** A client that stops acking while the backlog grows is presumed gone; the link then
  drops the connection, refuses that client's questions and puts the human back to "nobody to ask".
  Judging by silence needs a number that arrives regularly, and this is it.
* **coalescing.** Because the client says what it has, the link knows how far behind it is and may
  fold what is queued into fewer, larger messages. Without an ack there is no way to tell "slow" from
  "gone", and both answers would be guesses.
"""

from __future__ import annotations

from msgspec import Struct


class Ack(Struct, frozen=True, kw_only=True):
    """Everything up to and including ``seq`` has been rendered; nothing before it needs replaying."""

    seq: int = 0
