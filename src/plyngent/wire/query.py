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

from plyngent.wire.event import Entry  # noqa: TC001 — a `Struct` field's annotation is resolved here


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
    """One question a client asks. The number it carries is the link's, and it comes back on the reply.

    ``workspace`` names the tree the question is about, the same word a command uses and with the same
    meaning: empty is the tree this connection declared. A read about a tree this kernel does not serve
    answers with the empty value of its own kind — a read has no lane to refuse on, and "there is nothing
    there" is the truth about a tree nobody opened (E154).
    """

    id: int = 0
    workspace: str = ""


class Grants(Query, tag="grants", frozen=True, kw_only=True):
    """Which paths outside its workspace this conversation may reach, and how far on each.

    The read half of the one question a person answers about a path: what was allowed lives in the
    session's settings, and until this asked for it nothing showed it, so a "yes" was a fact a person
    had to remember (E134).
    """

    session: str = ""


class Trajectory(Query, tag="trajectory", frozen=True, kw_only=True):
    """Every row of one conversation, in the order it was written.

    A projection answers what a model is given — one chain, and only the records a model reads. This
    answers the other question: what happened here, in what order, and what continues what. Each row
    carries its parent, so the branches are in the answer and a client does not have to ask for them
    one at a time, and the row nothing continues is where a turn may grow.

    It is about a conversation, which is why it is the first read with a
    list in it: nothing here is a running total the kernel could have said in a flag.
    """

    session: str = ""


class Turns(Query, tag="turns", frozen=True, kw_only=True):
    """Which turns of one conversation are still running, right now.

    A read, and not something a client can work out for itself: a turn is started by a command and ends
    by an event, so a client that wants to know whether the work it asked for is over has two facts to
    compare and no way to be sure it has seen both. The kernel has one answer to that — the turns it is
    running — and it is a snapshot: an empty answer means this conversation is idle,
    and a client that wants to be told when that changes asks again after the next event.

    Empty is an answer and not a refusal, the same way an empty forest is: a conversation nothing is
    running in is the ordinary case, and it is the answer the lane this serves is waiting for.
    """

    session: str = ""


class Skills(Query, tag="skills", frozen=True, kw_only=True):
    """Which skills this host can see, and what is wrong with the ones that are not clean.

    A read, because it is about the *program's* environment: the directories were chosen when this
    kernel started, and a person asking "what does
    this thing know how to do" is asking about the machine in front of them, not about the turn.

    Nothing is session-shaped here, which is why the query carries no name: skills are found once at
    install, and every conversation sees the same table (``skill_list`` says the same thing to a model).
    """


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


class Forest(Reply, tag="forest", frozen=True, kw_only=True):
    """The answer to :class:`Trajectory`: one conversation's rows, parents and all.

    Empty is an answer and not a refusal: a conversation nothing has been written in yet holds no rows,
    and so does one this kernel is not serving — which is how a person learns there is nothing to walk.

    ``unfinished`` names the turns that have rows here and no row saying they ended. It is a value and not
    a verdict: a turn that is still running has no end row either, and only the kernel's own answer at this
    moment (``LiveTurns``) tells the two apart — so the reading of it belongs to whoever can see both.
    """

    entries: tuple[Entry, ...] = ()
    unfinished: tuple[int, ...] = ()


class LiveTurn(Struct, frozen=True, kw_only=True):
    """One turn that is running: which one, what kind, and the entry it is writing after.

    ``kind`` is text, for the reason every closed set kept in a row is: an answer
    written by today's build stays readable when the set grows a member. ``begin_after`` is the entry
    the turn continues, so a client that asks while a turn is live can say where it will go next.
    """

    turn: int = 0
    kind: str = ""
    begin_after: str | None = None


class LiveTurns(Reply, tag="live-turns", frozen=True, kw_only=True):
    """The answer to :class:`Turns`: the turns running in this conversation, oldest first.

    No count and no flag beside the list, because the list is the answer: "nothing is running" is the
    empty tuple, which a client reads the same way it reads a grant table with no rows.
    """

    turns: tuple[LiveTurn, ...] = ()


class SkillRow(Struct, frozen=True, kw_only=True):
    """One skill a person can see: its name, which root answered, where it lives, and what it says.

    ``root`` is the label the discovery uses (``plyngent-user``, ``claude-project``), because two copies
    of one name are two different directories and a listing that did not say which would be ambiguous.
    ``issues`` is what could not be read as it was written — a declared name that is not one, a missing
    description, a block that did not parse — carried as data, so the reader says it.
    """

    name: str = ""
    root: str = ""
    path: str = ""
    description: str = ""
    issues: tuple[str, ...] = ()


class ShadowedSkill(Struct, frozen=True, kw_only=True):
    """A name more than one root holds: the copy that wins, and the ones it wins over."""

    name: str = ""
    winner: str = ""
    copied: tuple[str, ...] = ()


class MissingRoot(Struct, frozen=True, kw_only=True):
    """A root with no directory: where it would have been, and which one it was."""

    path: str = ""
    label: str = ""


class SkillTable(Reply, tag="skill-table", frozen=True, kw_only=True):
    """The answer to :class:`Skills`: the rows, the shadows, and the roots nothing was found under.

    Empty is an answer and not a refusal — a host with no skills has none, and saying so is how a person
    learns there is nothing to read — and all three lists are in the one value because they are one
    question: what is here, what is here twice, and where there was nothing to look in.
    """

    skills: tuple[SkillRow, ...] = ()
    shadowed: tuple[ShadowedSkill, ...] = ()
    absent: tuple[MissingRoot, ...] = ()


type AnyQuery = Grants | Trajectory | Skills | Turns
type AnyReply = GrantTable | Forest | SkillTable | LiveTurns
