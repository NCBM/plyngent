"""What a client asks the kernel to tell it, and what it is told back.

A command is an intent and a read is a question, and until now only the first could cross a connection:
every lane ran one way with nothing to correlate, because nothing ever expected an answer. A read needs
one, so it travels as a query with a number on it and comes back as a reply carrying the same number —
the shape :class:`~plyngent.wire.ask.AnyAsk` already has in the other direction, and for the same
reason: two parties, one connection, and no second way to say which answer belongs to which question.

**A reply is the same kind as the query.** A ``Grants`` is answered by a ``GrantTable`` and by nothing
else, because a table is what "which paths may this conversation reach" *is*. Nothing here is an
envelope with a status and a payload, for the reason the ask lane gives: a value that could be any of
four things makes every reader ask which one it has.

**A query is never a command.** Commands change the world and are answered by events on the event lane,
which is how a client learns what came of one. A query changes nothing and is answered on its own lane,
where the answer cannot be folded, dropped, or overtaken: a read whose answer went missing would be a
frontend waiting forever, and the event lane is allowed to do exactly that to what it carries.

``id`` is the connection's own number and is stamped by the link, never by a caller: a client that could
choose it could answer somebody else's question.
"""

from __future__ import annotations

from msgspec import Struct


class Unanswered(Exception):  # noqa: N818 — a read nobody answered is a state, not a failed call
    """A read whose connection ended before the kernel answered it.

    Raised where a client is waiting for a reply, and never carried across a lane: what travels is the
    question and its number, and this is what this end says when the reply lane ends first. There is one
    message because there is one meaning — no answer is coming — and a caller that needs to know why
    looks at whether its connection is still there.
    """

    def __init__(self) -> None:
        super().__init__("this connection ended before the kernel answered")


class Query(Struct, frozen=True, kw_only=True, tag_field="query"):
    """One question a client asks. The number it carries is the link's, and it comes back on the reply."""

    id: int = 0


class Grants(Query, tag="grants", frozen=True, kw_only=True):
    """Which paths outside its workspace this conversation may reach, and how far on each.

    The read half of the one question a person answers about a path: what was allowed lives in the
    session's settings, and until this asked for it nothing showed it, so a "yes" was a fact a person
    had to remember (E134).
    """

    session: str = ""


class Reply(Struct, frozen=True, kw_only=True, tag_field="reply"):
    """What the kernel answers one query with: the same number, back on the lane a client is waiting on."""

    id: int = 0


class Reached(Struct, frozen=True, kw_only=True):
    """One path a conversation may reach, as the two words a person needs: which, and how far."""

    #: Written the way the path tools write a path: relative to the workspace where that is shorter.
    path: str = ""
    #: The access mode's own word — the same one the grant was asked with.
    mode: str = ""


class GrantTable(Reply, tag="grant-table", frozen=True, kw_only=True):
    """The answer to :class:`Grants`: every path this conversation holds, in the order they are kept.

    Empty is an answer and not a refusal: a conversation that has been allowed nothing outside its
    workspace is the ordinary case, and saying "nothing" is how a person learns there is nothing to
    take back.
    """

    reached: tuple[Reached, ...] = ()


type AnyQuery = Grants
type AnyReply = GrantTable
