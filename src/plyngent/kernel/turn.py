"""One turn, run: four stages between what a person said and what the journal holds.

``assemble → call → dispatch → settle``, in that order, over one :class:`TurnState`. Each stage is one
method and does one thing, which is the whole point of writing it this way rather than as one loop with
conditionals: a question about a request is asked in :meth:`~TurnRunner.assemble`, a question about an
answer in :meth:`~TurnRunner.dispatch`, and what happens when somebody interrupts is decided once, in
:meth:`~TurnRunner.run`.

Nothing else writes to the journal during a turn. The person's line, the model's answer and every tool
result go through :meth:`~TurnRunner._append`, so "what the history holds" has one author and a turn
that stops halfway still leaves a chain somebody can resume from.

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

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.model import Assistant, Reply, Request, Result, User
from plyngent.kernel.render import render
from plyngent.wire.event import AssistantText, TurnFinished, TurnStarted

if TYPE_CHECKING:
    from plyngent.kernel.catalog import ToolCatalog
    from plyngent.kernel.executor import ToolExecutor
    from plyngent.kernel.model import AnyMessage
    from plyngent.kernel.session import TurnCtx
    from plyngent.wire.event import AnyEvent

#: Why a turn that ended without being cancelled did not run to its own end.
FAILED = "the turn failed"


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

    def __init__(self, ctx: TurnCtx, *, catalog: ToolCatalog, executor: ToolExecutor) -> None:
        self._ctx = ctx
        self._catalog = catalog
        self._executor = executor
        self._calls = 0

    async def run(self, text: str) -> None:
        """Run the turn to its end, which is an answer, a refusal, or somebody stopping it.

        The three ends are all one path: the loop leaves when a reply asked for nothing, a cancellation
        is caught and becomes the reason the turn finished, and the ``finally`` is what makes every end
        look the same to the outside — the work joined, the finish announced, the turn forgotten.
        """
        ctx = self._ctx
        session = ctx.session.session_id
        state = TurnState(ctx=ctx, cursor=ctx.begin_after, text=text)
        self._emit(TurnStarted(session=session, turn=ctx.turn_id, begin_after=ctx.begin_after, kind=ctx.kind))
        reason = ""
        try:
            state = await self._append(state, User(text=text))
            while True:
                request = await self.assemble(state)
                reply = await self.call(state, request)
                state = await self.dispatch(state, reply)
                if not state.more:
                    break
                state = replace(state, ctx=state.ctx.next_round())
        except Cancelled as stopped:
            reason = str(stopped.reason)
        except Exception:
            # A turn that raised is a bug, and the turn still has to stop being the session's live one:
            # every later `Submit` is refused while a turn is live, so a turn that vanished would take
            # the session with it.
            reason = FAILED
            raise
        finally:
            _ = await ctx.cancel.settle(ctx.turn_address())
            self._emit(
                TurnFinished(session=session, turn=ctx.turn_id, reason=reason, ends_at=state.cursor),
            )
            ctx.end()

    # -- the stages ----------------------------------------------------------------

    async def assemble(self, state: TurnState) -> Request:
        """What the model is asked: the conversation up to ``cursor``, and the tools in name order.

        The conversation is the journal's projection rather than a list this turn kept up to date, so a
        request is exactly "what the history says" and a second turn of the same session sees what the
        first one wrote without anything having to be handed over.

        The tools are the view this session has: the residents, plus the groups it has been shown, and
        the system line says what it is still missing. A view is built per *round* rather than per turn,
        because a group asked for in one round arrives in the next — a block that changed inside a round
        would be a prefix that changed between two requests for no reason a provider could cache (§10.5).
        """
        trail = state.ctx.host.trail
        messages = await trail.derive(state.ctx.session.session_id, state.cursor)
        view = self._catalog.view(state.ctx.session.disclosed)
        return Request(system=view.index(), messages=messages, tools=view.prompts())

    async def call(self, state: TurnState, request: Request) -> Reply:
        """Ask the model for the next message, under the round's own address.

        The guard is what makes an interruption reach a request that is in flight: the round's scope sits
        under the turn's, so a cancel of either stops this, and the request is cancelled instead of being
        left running while the turn says it finished.
        """
        return await state.ctx.round_scope().guard(state.ctx.host.model.respond(request))

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
        said = await self._append(state, Assistant(text=reply.text, calls=reply.calls))
        if reply.text:
            self._emit(AssistantText(session=ctx.session.session_id, turn=ctx.turn_id, text=reply.text))
        if not reply.calls:
            return replace(said, more=False)
        asked = [(self._addressing(), call) for call in reply.calls]
        results = await self._executor.batch(ctx, asked)
        for call, result in zip(reply.calls, results, strict=True):
            text = render(result)
            said = await self._append(said, Result(call=call.id, tool=call.tool, status=result.status, text=text))
        return replace(said, more=True)

    # -- the pieces ----------------------------------------------------------------

    async def _append(self, state: TurnState, payload: AnyMessage) -> TurnState:
        """Write one entry at the end of the chain, and answer the state that continues from it."""
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
