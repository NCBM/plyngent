"""One turn, run: four stages between what a person said and what the journal holds.

``assemble → call → dispatch → settle``, in that order, over one :class:`TurnState`. Each stage is one
method and does one thing, which is the whole point of writing it this way rather than as one loop with
conditionals: a question about a request is asked in :meth:`~TurnRunner.assemble`, a question about an
answer in :meth:`~TurnRunner.dispatch`, and what happens when somebody interrupts is decided once, in
:meth:`~TurnRunner.run`.

Nothing else writes to the journal during a turn. The person's line, the model's answer and every tool
result go through :meth:`~TurnRunner._append`, so "what the history holds" has one author and a turn
that stops halfway still leaves a chain somebody can resume from — and a line the session owes the model
goes through :meth:`~TurnRunner._notice`, which is the one row that has to announce where it landed.

Two rules this file is built around:

* **Every await is guarded.** A request, a tool call, and the round they run under are all addressed in
  the turn's cancellation scope, so the token is what stops them — not an ``asyncio.CancelledError`` from
  somewhere else in the process — and a stopped turn still reports its reason instead of vanishing.
* **A cancelled turn is settled, not abandoned.** The results of calls that were in flight are still owed
  to the history, so the turn waits for the work it started before it says it finished.

What is deliberately *not* here yet: hooks (``before_request`` / ``after_response`` / ``before_tool``).
The stages are the places they will attach, and each is a single method so that one can be added without
moving anything else — but a hook nobody installs is a signature somebody designs around, and nothing
installs one today.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from msgspec.structs import replace as replaced

from plyngent.kernel.budget import calibrated, estimate
from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.documents import picture_media, refuses
from plyngent.kernel.failure import Failure
from plyngent.kernel.model import Assistant, Call, Delta, ModelFailed, Part, Picture, Reply, Request, Result, User
from plyngent.kernel.notices import (
    ELIDED,
    LIMIT,
    TERMINALS,
    elided,
    elision_summary,
    lost_terminals,
    lost_terminals_summary,
)
from plyngent.kernel.render import render, render_part
from plyngent.kernel.toolresult import OUTCOMES, Document, ToolResult
from plyngent.kernel.trail import (
    BY_BUDGET,
    Attempt,
    Effect,
    Notice,
    Summary,
    TurnEnded,
    Withheld,
    as_message,
    transcript,
)
from plyngent.kernel.trail import Usage as UsageRow
from plyngent.kernel.words import counted
from plyngent.wire.ask import Selection, Severity, as_picked
from plyngent.wire.event import (
    AnswerCut,
    AssistantDelta,
    AssistantText,
    AssistantThought,
    Compacted,
    Noticed,
    Pruned,
    TurnFinished,
    TurnKind,
    TurnStarted,
    Usage,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from plyngent.kernel.catalog import ToolCatalog
    from plyngent.kernel.executor import ToolExecutor
    from plyngent.kernel.model import AnyMessage, ToolDecl
    from plyngent.kernel.notices import Pending
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.trail import AnyRecord, Entry
    from plyngent.wire.event import AnyEvent

#: Why a turn that ended without being cancelled did not run to its own end.
FAILED = "the turn failed"

#: The reason written on a checkpoint the budget asked for, so a reader of the log can tell it from a
#: person's prune without decoding who asked (:data:`plyngent.kernel.trail.BY_BUDGET` says that).
OUTGREW = "the conversation outgrew what this route can hold"

#: How many times a turn may ask the model before it stops, unless `[turn] rounds` says otherwise.
#: Zero is "this program never said", the reading ``Caps.context_window`` gets: a turn ends when the
#: model has nothing more to ask for, and nothing here counts them (E150).
DEFAULT_ROUNDS = 0

#: The two of the three answers to the round wall's question that are not another window: a person
#: saying "this turn, no more counting" and a person saying "not any more". The words are the question's
#: own options, so the screen and the answer cannot disagree about them.
NO_MORE_ROUNDS = "no more limit for this turn"
STOP_ROUNDS = "stop here"


class _WalledError(Exception):
    """The turn cannot go on: the conversation does not fit the room, or it has run as far as it may.

    Not a ``ModelFailed``: no request was made, so there is no attempt to write down — the turn ended
    before it asked anything. It becomes a ``LIMIT`` turn-end and one sentence (:meth:`TurnRunner.run`).
    """


def _said(payload: AnyRecord) -> bool:
    """Whether this row is something one of the two speakers said, rather than a line about the log.

    A summary and a reminder project to messages as well, and neither is a round's: one is the compressed
    past, which a budget that hid it would be hiding the beginning of the conversation to make room for
    its end, and the other is the harness speaking at a boundary it chose.
    """
    return isinstance(payload, User | Assistant | Result)


def _rounds(
    chain: Sequence[Entry],
    *,
    hidden: Collection[str],
    skipping: int,
) -> list[list[tuple[Entry, AnyMessage]]]:
    """The chain's still-shown messages, oldest first, grouped into the rounds that wrote them.

    A round is the unit a checkpoint works in, because it is the unit both readers already have: half of
    one is a question with no answer or an answer to nothing. ``hidden`` is what an earlier checkpoint
    took away, and ``skipping`` is the turn being run — a round that has not finished being written is
    not a round yet, and the person's line it began with is the question being answered.
    """
    rounds: list[list[tuple[Entry, AnyMessage]]] = []
    for row in reversed(chain):
        if row.id in hidden or row.turn == skipping or not _said(row.payload):
            continue
        message = as_message(row.payload)
        if message is None:  # pragma: no cover — every record `_said` keeps projects to a message
            continue
        if rounds and rounds[-1][0][0].turn == row.turn:
            rounds[-1].append((row, message))
        else:
            rounds.append([(row, message)])
    return rounds


@dataclass(frozen=True, slots=True)
class TurnState:
    """What a turn carries from one stage to the next.

    ``cursor`` is the entry the conversation ends at right now, and it is the only thing that moves as
    the turn goes on: a stage answers a *new* state rather than editing this one, so no stage can be
    surprised by a cursor a later stage replaced. ``more`` is whether another round is owed — a reply
    that asked for no tool ends the turn.
    """

    ctx: TurnCtx
    cursor: str | None
    text: str = ""
    more: bool = False


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """Where a turn ended, and whether it did anything while it ran.

    ``acted`` is "it asked for at least one tool", and it exists for the one reader that has to decide
    whether to start another turn: a goal turn that only talked changed nothing a second turn could carry
    on from, so asking again would be insisting rather than progress (§10.4, E146). ``reason`` is the row's
    own answer to "did it end on its own", because a turn that was stopped or that failed is a turn a
    person is done with.
    """

    ends_at: str
    acted: bool
    reason: str = ""


class TurnRunner:
    """One turn, from the person's line to the last entry it wrote.

    Built per turn and thrown away: it holds the two things a turn needs from the session it belongs to
    (the tools it may reach, and the executor that runs them) plus a counter for naming calls, and
    nothing that outlives the turn.
    """

    _ctx: TurnCtx
    _catalog: ToolCatalog
    _executor: ToolExecutor
    _calls: int
    #: How many rounds this turn has run, and how many it may before the wall — ``None`` for a turn with
    #: no wall at all. Both are this runner's alone: a window is a fact about one turn, so it is not a
    #: setting, not a row, and a goal turn started afterwards begins with a fresh one (E150).
    _ran: int
    _allowed: int | None
    #: Whether this round's thinking has already been said in pieces, so dispatch does not say it twice.
    _thought_said: bool

    def __init__(self, ctx: TurnCtx, *, catalog: ToolCatalog, executor: ToolExecutor) -> None:
        self._ctx = ctx
        self._catalog = catalog
        self._executor = executor
        self._calls = 0
        self._ran = 0
        self._allowed = ctx.host.rounds or None
        self._thought_said = False

    async def run(self, text: str) -> TurnEnd:
        """Run the turn to its end, which is an answer, a refusal, or somebody stopping it.

        The three ends are all one path: the loop leaves when a reply asked for nothing, a cancellation
        is caught and becomes the reason the turn finished, and the ``finally`` is what makes every end
        look the same to the outside — the work joined, what the session owes the model said, the finish
        announced, the turn forgotten.

        A ``GOAL`` turn writes no line of its own: nothing new was said, so what it reads comes from the
        prompt (:meth:`assemble`) and its history is the one the turn before it left (E146). A ``RETRY``
        turn is the same shape for the other reason: nobody said anything new, they asked the same
        question again, so it adds no message and asks the model against the chain as it stands (E158).
        """
        ctx = self._ctx
        session = ctx.session.session_id
        state = TurnState(ctx=ctx, cursor=ctx.begin_after, text=text)
        self._emit(TurnStarted(session=session, turn=ctx.turn_id, begin_after=ctx.begin_after, kind=ctx.kind))
        reason = ""
        failure: Failure | None = None
        ended = ""
        try:
            if ctx.kind is TurnKind.COMPACT:
                state = await self._summarized(state)
            else:
                if ctx.kind not in (TurnKind.GOAL, TurnKind.RETRY):
                    state = await self._append(state, User(text=text))
                while True:
                    await self._walled(state)
                    self._terminals_gone(state)
                    state = await self._told(state)
                    state, request = await self.assemble(state)
                    reply = await self.call(state, request)
                    state = await self.dispatch(state, reply)
                    if not state.more:
                        break
                    state = replace(state, ctx=state.ctx.next_round())
        except Cancelled as stopped:
            reason = str(stopped.reason)
        except _WalledError as walled:
            # Nothing was asked, so this is not an attempt: the turn ends having said why it could not
            # go on — the room it does not fit, or the rounds it has run (E149/E150).
            failure = Failure.LIMIT
            reason = str(walled)
        except ModelFailed as unmet:
            # A model that could not be asked is a *result*, not a bug: the turn ends having done what
            # it could, and says which kind of trouble it met so a person can act on it (and a program
            # can switch on it). Nothing is re-raised — a turn that raised would be handed to a task
            # nobody awaits, and the one place that prints is a terminal the kernel may not write to.
            failure = unmet.failure
            reason = unmet.says
            await self._attempted(state, unmet)
        except Exception:
            # A turn that raised is a bug, and the turn still has to stop being the session's live one:
            # every later `Submit` is refused while a turn is live, so a turn that vanished would take
            # the session with it.
            reason = FAILED
            raise
        finally:
            _ = await ctx.cancel.settle(ctx.turn_address())
            state = await self._told(state)
            ended = await self._ended(state.cursor, reason=reason, failure=failure)
            self._emit(
                TurnFinished(
                    session=session,
                    turn=ctx.turn_id,
                    reason=reason,
                    ends_at=ended,
                    failure="" if failure is None else str(failure),
                ),
            )
            ctx.end()
        return TurnEnd(ends_at=ended, acted=self._calls > 0, reason=reason)

    # -- the round wall -------------------------------------------------------------
    async def _walled(self, state: TurnState) -> None:
        """Count one round, and at the wall either open another window or end the turn there.

        A turn that keeps asking is a turn nothing inside it stops: the model asks for a tool, the tool
        answers, so the loop goes round again, and the only thing that could say "not this many times" is
        a number somebody chose. `[turn] rounds` is that number and ``0`` is nobody having chosen one, so
        an unconfigured program is the program it always was.

        The wall has two shapes. Hard, it ends the turn — ``LIMIT``, the same member a conversation that
        cannot be fitted uses, because it is the same kind of fact. Soft, it asks the person first, and
        what came back is the new window: another ``rounds``, no limit for this turn at all, or stop. The
        window is this runner's, so nothing about it is written down: it is a fact about one turn, and a
        goal turn started afterwards is a new turn with a new one (E150).
        """
        limit = state.ctx.host.rounds
        allowed = self._allowed
        if limit and allowed is not None and self._ran >= allowed:
            if not state.ctx.host.soft:
                msg = (
                    f"this turn ran {counted(self._ran, 'round')}, which is the "
                    f"{counted(limit, 'round')} this program runs before it stops"
                )
                raise _WalledError(msg)
            self._allowed = await self._asked_for_more(state, allowed, limit)
        self._ran += 1

    async def _asked_for_more(self, state: TurnState, allowed: int, limit: int) -> int | None:
        """Ask whether this turn may keep going, and answer the window that came back — or end the turn.

        Three answers, and the third one is why this is a choice rather than a yes-or-no question: a
        person watching a turn that is getting somewhere says "no more counting" and means it, while a
        person who wants to look at what it did says stop. The question is asked at the lightest level
        with its own default in front of the person, so a client left to answer by itself repeats the
        option an empty line would have given — which is what keeps an unattended run inside the ceiling
        it was given rather than turning the wall into a checkpoint (E150).

        Nobody answering is the wall: an answer with no option in it is not "again", and a question put
        to a client that is not there is never permission to go on.
        """
        again = f"another {counted(limit, 'round')}"
        answer = as_picked(
            await state.ctx.ask(
                Selection(
                    prompt=(
                        f"this turn has run {counted(self._ran, 'round')}, which is the "
                        f"{counted(limit, 'round')} this program asks about — keep going?"
                    ),
                    options=(again, NO_MORE_ROUNDS, STOP_ROUNDS),
                    default=again,
                    severity=Severity.ROUTINE,
                ),
                tool="",
                call="rounds",
            )
        )
        match answer.value:
            case _ if answer.value == again:
                return allowed + limit
            case _ if answer.value == NO_MORE_ROUNDS:
                return None
            case _ if answer.value == STOP_ROUNDS:
                msg = f"this turn was stopped after {counted(self._ran, 'round')}, which is where it asked"
                raise _WalledError(msg)
            case _:
                msg = (
                    f"this turn ran {counted(self._ran, 'round')}, and nobody said to go past the "
                    f"{counted(limit, 'round')} this program asks about"
                )
                raise _WalledError(msg)

    def _terminals_gone(self, state: TurnState) -> None:
        """Say the terminals a replaced broker took, if this conversation lost any (E151).

        Taken at the top of a round, before anything is owed to the model, because it belongs with the
        other things said at a boundary: what a model needs to know before it chooses its next call. It is
        not a call that failed — the refusal for an old id still stands — it is the news that arrives
        before the model would have spent a call on finding out.

        Nothing is written here: this fires a sentence, and :meth:`_told` is the one place a fired sentence
        becomes a row.
        """
        gone = state.ctx.host.pty.take_lost(state.ctx.session.session_id)
        if gone:
            state.ctx.session.notices.say(TERMINALS, lost_terminals(gone), summary=lost_terminals_summary(gone))

    async def _told(self, state: TurnState) -> TurnState:
        """Say the lines this session owes the model, at a boundary where nothing is being asked.

        Two boundaries and one rule. The top of every round, before the request is assembled, so that the
        model reads on *this* turn's next request what this turn just changed — a plan updated by a call
        is what the next round is for. And the settle, before the end row, so ``turn-ended`` stays the
        last line of a turn while the projection still ends with the reminder.
        model reads on *this* turn's next request what this turn just changed — a plan updated by a call
        is what the next round is for. And the settle, before the end row, so ``turn-ended`` stays the
        last line of a turn while the projection still ends with the reminder.

        Never in the middle of a round: a row appended between an answer and its results would sit inside
        the stretch a request was built from, and the next projection would read a conversation the
        request that produced it never saw.
        """
        ctx = state.ctx
        for _key, pending in ctx.session.notices.owed():
            state = await self._notice(state, pending)
        return state

    async def _notice(self, state: TurnState, pending: Pending) -> TurnState:
        """Write one line the session owes the model, and say where it landed.

        Its own append rather than :meth:`_append` because it is the one row that announces itself: the id
        is what a person's line names, so it comes back from the same call that wrote it rather than being
        read off the state afterwards.
        """
        ctx = state.ctx
        entry = await ctx.host.trail.append(
            ctx.session.session_id,
            Notice(text=pending.body, summary=pending.summary),
            begin_after=state.cursor,
            turn=ctx.turn_id,
        )
        self._emit(Noticed(session=ctx.session.session_id, entry=entry.id, summary=pending.summary))
        return replace(state, cursor=entry.id)

    async def _ended(self, cursor: str | None, *, reason: str, failure: Failure | None) -> str:
        """Write down that this turn ended, and answer the entry a reader finds it at.

        A row and not only the event, for the reason a call's cost is a row: the event is for whoever is
        watching, and this is for whoever reads the log later. What it adds that nothing else has is that
        the turn *said* it ended — a journal whose rows simply stop cannot tell a turn that finished from
        a process that was killed in the middle of one, and the second is the case a person needs to know
        about, because nothing was settled.

        Below the last entry of the turn rather than beside the chain, because a turn's end is a place the
        next message really does continue from. Nothing can be written after it: this is the last thing the
        turn does, which is what makes the row worth trusting as an answer — and it is why the cursor is a
        parameter rather than read off the state: what the session owed the model is written above it.
        """
        ctx = self._ctx
        entry = await ctx.host.trail.append(
            ctx.session.session_id,
            TurnEnded(turn=ctx.turn_id, reason=reason, failure="" if failure is None else str(failure)),
            begin_after=cursor,
            turn=ctx.turn_id,
        )
        return entry.id

    # -- the compaction -------------------------------------------------------------
    async def _summarized(self, state: TurnState) -> TurnState:
        """Ask the model for a summary of the stretch this turn was pointed at, and write that one row.

        A compaction is a turn because the request is one: it is cancelled like any other, it costs what
        it costs, and an endpoint that could not be reached leaves an attempt rather than nothing. What it
        does *not* do is dispatch — a summary request carries no tools, and an answer that asked for one
        anyway is not something this turn can act on.

        The operator's line is a **directive, not a message**: the summary stands for the stretch below it,
        so a user row reading "please summarise this" would be a thing the model is told happened.

        The one request that cannot be checkpointed is this one. A compaction *is* what makes room, so
        putting a stretch aside to make room for it would be answering the question with itself; and its
        whole input is the conversation, so it can be too big before anything is asked. What is left is to
        cut the input and say so — the summary then begins partway through, and knows it
        (:meth:`_transcript`).
        """
        ctx = state.ctx
        system, instruction = ctx.host.prompt.summarize(state.text)
        shown = list(await self._seen(ctx, await ctx.host.trail.derive(ctx.session.session_id, state.cursor)))
        room = self._room(state)
        shown, left = self._transcript(state, shown, system=system, instruction=instruction, room=room)
        if left:
            system, instruction = ctx.host.prompt.summarize(state.text, dropped=left)
        asked = ctx.host.routes.resolve(ctx.session.route)
        request = Request(
            system=system,
            messages=(*shown, User(text=instruction)),
            tools=(),
            route="" if asked is None else asked.name,
            cache_key=ctx.session.cache_key,
        )
        if room and self._measure(state, request.messages, system=request.system) > room:
            msg = (
                f"a summary of this conversation needs {self._measure(state, request.messages, system=request.system)} "
                f"tokens and this route holds {room}, which is more than the instruction alone"
            )
            raise _WalledError(msg)
        reply = await self.call(state, request)
        text = reply.text.strip()
        if not text:
            msg = "the model's summary was empty, so nothing was written"
            raise ModelFailed(Failure.PROTOCOL, msg)
        entry = await ctx.host.trail.append(
            ctx.session.session_id,
            Summary(text=text, replaces=await self._replaced(ctx.session.session_id, state.cursor)),
            turn=ctx.turn_id,
        )
        _ = await ctx.session.note_history()
        self._emit(Compacted(session=ctx.session.session_id, entry=entry.id, chars=len(text)))
        return replace(state, cursor=entry.id, more=False)

    async def _replaced(self, session: str, cursor: str | None) -> tuple[str, str] | None:
        """The stretch a summary stands for: the first and last rows of the chain it was given.

        ``replaces`` is bookkeeping a person reads ("this text stands for X..Y"), so it is worked out from
        the chain rather than guessed at: the two ends of the very rows the request carried.
        """
        if cursor is None:
            return None
        rows = {row.id: row for row in await self._ctx.host.trail.entries(session)}
        chain: list[str] = []
        here: str | None = cursor
        while here is not None and here in rows:
            chain.append(here)
            here = rows[here].begin_after
        if not chain:
            return None
        return chain[-1], chain[0]

    # -- the stages ----------------------------------------------------------------

    async def assemble(self, state: TurnState) -> tuple[TurnState, Request]:
        """What the model is asked: the conversation up to ``cursor``, and the tools in name order.

        The conversation is the journal's projection rather than a list this turn kept up to date, so a
        request is exactly "what the history says" and a second turn of the same session sees what the
        first one wrote without anything having to be handed over.

        The tools are the view this session has: the residents, plus the groups it has been shown, and
        the prompt the host was installed with says what that view means — the playbook of every group it
        can reach, and the index line naming the ones it cannot. A view is built per *round* rather than
        per turn, because a group asked for in one round arrives in the next — a block that changed inside
        a round would be a prefix that changed between two requests for no reason a provider could cache
        (§10.5).

        ``route`` and ``cache_key`` are filled here for the same reason the tools are: both are facts
        about *this* request. Which route answers is read at this boundary, so a ``SetRoute`` lands on the
        next round rather than on a turn that is half way through, and the key names the prefix by the
        session and the two counters that say where its prompt stands.

        The answer is the *state* as well as the request, because fitting a conversation to its route may
        write a row (:meth:`_fitted`): a checkpoint hides a stretch, so what the turn continues from is no
        longer what it was handed.
        """
        ctx = state.ctx
        view = self._catalog.view(ctx.session.disclosed)
        asked = ctx.host.routes.resolve(ctx.session.route)
        system = ctx.host.prompt.compose(view, continuing=self._carrying_on(state))
        tools = view.prompts()
        state, messages = await self._fitted(state, system=system, tools=tools)
        return state, Request(
            system=system,
            messages=messages,
            tools=tools,
            route="" if asked is None else asked.name,
            cache_key=ctx.session.cache_key,
        )

    async def _fitted(
        self,
        state: TurnState,
        *,
        system: str,
        tools: tuple[ToolDecl, ...],
    ) -> tuple[TurnState, tuple[AnyMessage, ...]]:
        """The conversation this request may carry: all of it, or all of it that still fits.

        A conversation has no size until somebody names a window, and only a route may: ``0`` is "this
        endpoint never said", and a denominator nobody stated is not a small one, so an unbudgeted route
        is sent its whole history however long it is.

        When the room is stated and the request outgrows it, the oldest **whole rounds** are put aside —
        a checkpoint, not a truncation (§10.4, E149). Three things that are decisions rather than
        details: the cut is by round, because half a round is a question with no answer or an answer to
        nothing; it goes *deep*, to half the room, because every checkpoint moves the prefix a provider
        cached and a shallow cut would move it every round; and it is written down, so the same history
        is read the same way twice and the entries are still there to travel back to. The line the model
        reads about it is a notice the session owes, said here because this is a boundary where no
        request is in flight.

        What cannot be fitted even after that is the one case this refuses instead of guessing: a
        conversation whose newest round alone is too big ends the turn with ``LIMIT`` rather than being
        sent to a provider that would reject it, and rather than being pruned to nothing.
        """
        room = self._room(state)
        messages = await self._shown(state)
        size = self._measure(state, messages, system=system, tools=tools)
        if not room or size <= room:
            return state, messages
        taken = await self._put_aside(state, room=room, size=size)
        if not taken:
            msg = (
                f"this conversation needs {size} tokens and this route holds {room}, and nothing in it is "
                "old enough to put aside"
            )
            raise _WalledError(msg)
        targets = tuple(row.id for row, _message in taken)
        kept = await self._kept(state, [message for _row, message in taken])
        ctx = state.ctx
        entry = await ctx.host.trail.append(
            ctx.session.session_id,
            Withheld(targets=targets, reason=OUTGREW, by=BY_BUDGET),
            begin_after=state.cursor,
            turn=ctx.turn_id,
        )
        # Said outward as well as written: a client's own cursor is where its next message continues from,
        # so a prune nobody announced is a stretch every other frontend would read back by growing from
        # below it (E53's bug, second case).
        self._emit(Pruned(session=ctx.session.session_id, entry=entry.id, targets=targets, reason=OUTGREW))
        ctx.session.notices.say(
            ELIDED,
            elided(len(targets), by=BY_BUDGET, kept=kept),
            summary=elision_summary(len(targets), by=BY_BUDGET),
        )
        # The elision is the second number a provider's prefix cache keys on: what the next prompt holds
        # has changed, and a key that did not move would claim a prefix nobody answered for (§10.5).
        _ = await ctx.session.note_history()
        state = await self._told(replace(state, cursor=entry.id))
        messages = await self._shown(state)
        left = self._measure(state, messages, system=system, tools=tools)
        if left > room:
            msg = (
                f"this conversation still needs {left} tokens and this route holds {room}, even after "
                f"putting {len(targets)} entries aside"
            )
            raise _WalledError(msg)
        return state, messages

    def _measure(
        self,
        state: TurnState,
        messages: Sequence[AnyMessage],
        *,
        system: str = "",
        tools: Sequence[ToolDecl] = (),
    ) -> int:
        """How many tokens this request will be counted as, corrected by what the last one actually was.

        The character estimate is a rule of thumb; the provider's own count of the payload it was just
        handed is the correction, and :func:`~plyngent.kernel.budget.calibrated` applies it. A session that
        has asked nothing yet — or whose route just moved — has no correction and is measured as it is:
        wrong by one request, rather than wrong until somebody says so.
        """
        measured = estimate(messages, system=system, tools=tools)
        return calibrated(measured, state.ctx.session.calibration)

    def _room(self, state: TurnState) -> int:
        """How many tokens one request may be, or ``0`` when this route never said what it can hold.
        The room the answer will need is taken off the top: ``max_output`` is what a request is given for
        what it is about to say, and a budget that forgot it would fill the window exactly and then be cut
        off mid-sentence. A route whose window cannot even hold its own answer is a route whose numbers do
        not describe it, and nothing is invented from them: it is treated as one that never said a window.
        """
        asked = state.ctx.host.routes.resolve(state.ctx.session.route)
        if asked is None or asked.caps.context_window <= 0:
            return 0
        room = asked.caps.context_window - asked.caps.max_output
        return max(0, room)

    async def _put_aside(
        self,
        state: TurnState,
        *,
        room: int,
        size: int,
    ) -> tuple[tuple[Entry, AnyMessage], ...]:
        """Which entries to put aside, and what they said: whole rounds, oldest first, to half the room.

        Read off the chain every time rather than remembered, because "what is still shown" is a fact
        about the log: an earlier checkpoint already hid a stretch, and what this one may hide is what
        that one left.

        Half the room is the target rather than the wall itself: a conversation pruned exactly to the edge
        would be back at the wall within one answer, and every visit writes a row and moves a prefix.
        """
        ctx = state.ctx
        rows = {row.id: row for row in await ctx.host.trail.entries(ctx.session.session_id)}
        chain: list[Entry] = []
        here = state.cursor
        while here is not None and here in rows:
            chain.append(rows[here])
            here = rows[here].begin_after
        hidden = {target for row in chain if isinstance(row.payload, Withheld) for target in row.payload.targets}
        wanted, freed = size - room // 2, 0
        taken: list[tuple[Entry, AnyMessage]] = []
        for round_ in _rounds(chain, hidden=hidden, skipping=ctx.turn_id):
            if freed >= wanted:
                break
            for row, message in round_:
                taken.append((row, message))
                freed += self._measure(state, (message,))
        return tuple(taken)

    def _transcript(
        self,
        state: TurnState,
        messages: list[AnyMessage],
        *,
        system: str,
        instruction: str,
        room: int,
    ) -> tuple[list[AnyMessage], int]:
        """The newest part of a conversation that fits the room a summary is sent to, and how many were cut.

        Cut from the oldest, because a summary that begins partway through is still a summary of the part
        somebody is working on, while one that stops short is a summary of nothing anyone needs. Nothing is
        written down here and nothing is hidden: a compaction is already the last request a stretch is ever
        sent in, so what it leaves out is not kept from a later turn — it is simply not read aloud.
        """
        if not room:
            return messages, 0
        sizes = [self._measure(state, (message,)) for message in messages]
        total = self._measure(state, (*messages, User(text=instruction)), system=system)
        left = 0
        while left < len(messages) and total > room:
            total -= sizes[left]
            left += 1
        return messages[left:], left

    async def _kept(self, state: TurnState, messages: Sequence[AnyMessage]) -> str:
        """The stretch as one document, and the line that names it — or nothing when it cannot be kept.

        A budget puts things aside; it does not decide they are worth forgetting. So the text goes into
        the store beside the journal, as the transcript it was read from, and the line the model is handed
        is the one a handle always is — the call that reads it back is the one every other kept body
        arrives with (E56/E149).

        Nothing is kept when the store would refuse it: a body past the store's own limit is a body no
        handle can stand for, and the sentence then says what is true instead of promising a handle.
        """
        ctx = state.ctx
        text = transcript(messages)
        if refuses(text) is not None:
            return ""
        handle = await ctx.host.documents.put(
            ctx.session.session_id,
            text,
            summary=f"{counted(len(messages), 'earlier entry')} put aside",
        )
        return render_part(Document(document_id=handle.id, summary=handle.summary, chars=handle.chars))

    async def _shown(self, state: TurnState) -> tuple[AnyMessage, ...]:
        """The conversation as this route can read it, pictures and all — the value a request carries."""
        ctx = state.ctx
        return await self._seen(ctx, await ctx.host.trail.derive(ctx.session.session_id, state.cursor))

    def _carrying_on(self, state: TurnState) -> str:
        """The section a turn the kernel started itself carries, and nothing for any other turn.

        A goal turn has no message of its own — nobody said anything — so what it is asked to do can only
        live in the prompt, and it is the one section that changes from one request to the next (E146).
        """
        if state.ctx.kind is not TurnKind.GOAL:
            return ""
        return state.ctx.host.prompt.carrying_on(state.ctx.session.todos.reminder())

    async def _seen(self, ctx: TurnCtx, messages: tuple[AnyMessage, ...]) -> tuple[AnyMessage, ...]:
        """The conversation as this route can read it: a picture travels only where both sides can take it.

        Two gates, and both are needed. The *route's* ``vision`` says what the model on the other end can
        read; the *dialect's* ``pictures`` says whether the wire has a channel for one at all — a
        text-only dialect answers a tool with text, and putting pixels in a person's message would be
        inventing a speaker. Where either says no, the handle stays in the words and nothing is fetched:
        the model reads what the tool said, which is true either way.

        The bytes are fetched here rather than stored in the row, which is what keeps "a body never enters
        an entry" true while a model still sees an image (E56).
        """
        asked = ctx.host.routes.resolve(ctx.session.route)
        codec = None if asked is None else asked.codec()
        if asked is None or not asked.caps.vision or codec is None or not codec.pictures:
            return messages
        seen: list[AnyMessage] = []
        for message in messages:
            if isinstance(message, Result) and message.documents:
                seen.append(await self._fetched(ctx, message))
            else:
                seen.append(message)
        return tuple(seen)

    async def _fetched(self, ctx: TurnCtx, message: Result) -> Result:
        """One tool result with the pictures among its handles filled in, and unchanged when there are none.

        A handle that is not a picture — a truncated body — is *not* attached: the whole point of the
        separation is that a handle says where content is and the media type says what it is, so only the
        ones the bytes identify as an image travel as an image.
        """
        pictures: list[Picture] = []
        for handle in message.documents:
            found = await ctx.host.documents.get(ctx.session.session_id, handle)
            if found is None:
                continue
            media = picture_media(found.data)
            if media is not None:
                pictures.append(Picture(media=media, data=found.data))
        return replaced(message, pictures=tuple(pictures)) if pictures else message

    async def call(self, state: TurnState, request: Request) -> Reply:
        """Ask the model for the next message, under the round's own address.

        The guard is what makes an interruption reach a request that is in flight: the round's scope sits
        under the turn's, so a cancel of either stops this, and the request is cancelled instead of being
        left running while the turn says it finished.

        What the asking cost is said outward and written down here rather than by the model: a model
        answers a value, and every event on this lane is the kernel's to emit. An answer that stopped short
        is said the same way, because both are facts about the request that just came back and neither is
        the turn's end.

        What the model *is saying* is said outward while it is still saying it — one event per fragment,
        as they arrive — but the answer this returns is the whole one, because that is what the turn
        writes down (§0.13, E94). A model that cannot stream says one thing and no fragments, so this is
        one loop and not two.
        """
        reply = await state.ctx.round_scope().guard(self._answered(state, request))
        if reply.usage.reported:
            await self._cost(state, reply, request)
        if reply.cut is not None:
            self._emit(
                AnswerCut(session=state.ctx.session.session_id, turn=state.ctx.turn_id, reason=str(reply.cut)),
            )
            self._cut_short(state, reply.cut)
        return reply

    def _cut_short(self, state: TurnState, cut: Failure) -> None:
        """Leave the model a line about the answer it gave that stopped at the limit.

        A truncated answer is silent from the model's side: the next request carries half a thought as its
        own words, and nothing in the history says why it stops in the middle — so the model goes on as if
        it had finished. One reminder fixes that, written at the next boundary (E142), because that is the
        first moment a request is not in flight.

        Only a limit is said. A cut the provider chose — a content filter, a refusal — is not "there is
        more of this to say", and telling a model to carry on from a sentence it was stopped from saying
        would be this kernel giving advice about somebody else's decision.
        """
        if cut is not Failure.LIMIT:
            return
        state.ctx.session.notices.say(
            LIMIT,
            "the answer you gave last stopped at the output limit, so it ended mid-thought",
            summary="the last answer hit the output limit",
        )

    async def _cost(self, state: TurnState, reply: Reply, request: Request) -> None:
        """Write down what this request cost, and what it was thought to cost, and then say it.

        The row comes first because the two are different readers: the event is the line a person sees
        while the turn is running, and the row is what makes "what has this conversation spent" a question
        that can be asked after the process that spent it is over. E133 keeps the price out of the kernel,
        so the three counts and the model that charged them are what is written — an outside reader adds
        up whatever a price list says, and cannot if the row does not name the model.

        ``estimate`` is of the very ``request`` this reply answers, which is the only payload its
        ``prompt`` count could belong to: pairing it with anything else would not measure this build's
        error but the difference between two decks of cards (§10.5, E149).
        """
        ctx = state.ctx
        usage = reply.usage
        sent = estimate(request.messages, system=request.system, tools=request.tools)
        _ = await ctx.host.trail.append(
            ctx.session.session_id,
            UsageRow(
                turn=ctx.turn_id,
                route=ctx.session.route,
                epoch=ctx.session.route_epoch,
                prompt=usage.prompt,
                completion=usage.completion,
                cached=usage.cached,
                estimate=sent,
            ),
            turn=ctx.turn_id,
        )
        # The same payload, counted twice: what this build said and what the provider charged. The ratio is
        # what the next decision is scaled by, so the session keeps it — and only while there is a charge
        # to compare against, since a route that reports nothing is not a route that charged nothing.
        if usage.prompt:
            ctx.session.calibration = (sent, usage.prompt)
        self._emit(
            Usage(
                session=ctx.session.session_id,
                turn=ctx.turn_id,
                prompt=usage.prompt,
                completion=usage.completion,
                cached=usage.cached,
            ),
        )

    async def _answered(self, state: TurnState, request: Request) -> Reply:
        """Every fragment of the answer said outward as it arrives, and then the answer itself.

        A fragment is said in the vocabulary its half of the answer belongs to — words as
        :class:`~plyngent.wire.event.AssistantDelta`, thinking as
        :class:`~plyngent.wire.event.AssistantThought` — because the two are two things a person reads
        differently and a client may keep one and drop the other (§10.4, E145). What was said in pieces
        is remembered for the round so that a whole thought is not written twice (:meth:`dispatch`).
        """
        async for piece in state.ctx.host.model.stream(request):
            if isinstance(piece, Delta):
                self._spoken(state, piece)
                continue
            return piece
        msg = "the model's stream ended without the answer it was adding up to"
        raise ModelFailed(Failure.PROTOCOL, msg)

    def _spoken(self, state: TurnState, piece: Delta) -> None:
        """One fragment outward, as the half of the answer it is a fragment of."""
        if not piece.text:
            return
        session, turn = state.ctx.session.session_id, state.ctx.turn_id
        if piece.part is Part.THOUGHT:
            self._thought_said = True
            self._emit(AssistantThought(session=session, turn=turn, text=piece.text))
            return
        self._emit(AssistantDelta(session=session, turn=turn, text=piece.text))

    async def dispatch(self, state: TurnState, reply: Reply) -> TurnState:
        """Write what the model answered, then run every call it asked for, as one batch.

        The answer is written before the first call runs, and that order is not a preference: a turn that
        stopped between the two would otherwise leave results whose question is nowhere in the history,
        and a model handed that history back would be reading answers to nothing.

        The calls run as far in parallel as the paths they hold allow
        (:meth:`~plyngent.kernel.executor.ToolExecutor.batch`), and their results are written in the order
        the model asked for them — one round has to leave one history, whichever call came back first.
        """
        ctx = state.ctx
        thought = self._thought_said
        self._thought_said = False
        if not thought:
            # A route that does not stream its thinking said none of it yet: what it thought is said here,
            # in one piece each, so a person reads it whether or not the route could send it early.
            for one in reply.thoughts:
                if one.text:
                    self._emit(AssistantThought(session=ctx.session.session_id, turn=ctx.turn_id, text=one.text))
        said = await self._append(state, Assistant(text=reply.text, calls=reply.calls, thoughts=reply.thoughts))
        if reply.text:
            self._emit(AssistantText(session=ctx.session.session_id, turn=ctx.turn_id, text=reply.text))
        if not reply.calls:
            return replace(said, more=False)
        asked = [(self._addressing(), call) for call in reply.calls]
        results = await self._executor.batch(ctx, asked)
        for call, result in zip(reply.calls, results, strict=True):
            said = await self._changed(said, call, result)
            text = render(result)
            said = await self._append(
                said,
                Result(call=call.id, tool=call.tool, status=result.status, text=text, documents=result.handles),
            )
        return replace(said, more=True)

    async def _changed(self, state: TurnState, call: Call, result: ToolResult) -> TurnState:
        """Write what this call left different, directly below the result it belongs to.

        Below and not above, and that is the whole of the ordering rule: the effect sits at the end of the
        chain the result ends, so the two are adjacent and neither names the other, while the row a turn
        continues from stays the result.

        A call that changed nothing writes nothing — the same rule a session's own settings follow — and a
        result produced without a store still records *what* changed with no version to read it back from,
        because "these paths are not what you last saw" is true whether or not this host keeps bytes.
        """
        effects = result.effects
        if not (effects.changed or effects.external or effects.approvals or effects.unswept):
            return state
        return await self._append(
            state,
            Effect(
                turn=state.ctx.turn_id,
                call=call.id,
                tool=call.tool,
                revision=effects.revision,
                changed=tuple(str(one) for one in effects.changed),
                external=effects.external,
                approvals=effects.approvals,
                unswept=effects.unswept,
            ),
        )

    # -- the pieces ----------------------------------------------------------------

    async def _attempted(self, state: TurnState, unmet: ModelFailed) -> None:
        """Write down that a request was made and never answered, and under which route.

        A **root**, deliberately: an attempt is not a place a conversation continues from, so it goes
        beside the chain rather than into it, and a turn that stopped here leaves the history it had
        rather than a node every later message would descend from. It is also the only record of the
        route *epoch* that was in force — the setting says where the session is now, and this says where
        it was when it tried (R4's "attempt is a first-class citizen").
        """
        ctx = state.ctx
        route = ctx.host.routes.resolve(ctx.session.route)
        _ = await ctx.host.trail.append(
            ctx.session.session_id,
            Attempt(
                turn=ctx.turn_id,
                route="" if route is None else route.name,
                epoch=ctx.session.route_epoch,
                failure=str(unmet.failure),
                retry=str(OUTCOMES[unmet.failure][1]),
                says=unmet.says,
            ),
            turn=ctx.turn_id,
        )

    async def _append(self, state: TurnState, payload: AnyRecord) -> TurnState:
        """Write one entry at the end of the chain, and answer the state that continues from it.

        ``payload`` is one *record*, which is wider than a message: the bookkeeping this turn writes — an
        attempt, an effect — goes down the same road, so "what the history holds" still has one author.
        """
        ctx = state.ctx
        trail = ctx.host.trail
        entry = await trail.append(ctx.session.session_id, payload, begin_after=state.cursor, turn=ctx.turn_id)
        return replace(state, cursor=entry.id)

    def _addressing(self) -> str:
        """The segment of the turn's scope the next call is named by — ``tool-1``, ``tool-2``, …."""
        self._calls += 1
        return f"tool-{self._calls}"

    def _emit(self, event: AnyEvent) -> None:
        """Say something outward through the host's human channel, whether or not anybody is attached."""
        self._ctx.host.human.emit(event)
