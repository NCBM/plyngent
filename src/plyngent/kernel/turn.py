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

from msgspec.structs import replace as replaced

from plyngent.kernel.cancel import Cancelled
from plyngent.kernel.documents import picture_media
from plyngent.kernel.failure import Failure
from plyngent.kernel.model import Assistant, Call, Delta, ModelFailed, Picture, Reply, Request, Result, User
from plyngent.kernel.render import render
from plyngent.kernel.toolresult import OUTCOMES, ToolResult
from plyngent.kernel.trail import Attempt, Effect
from plyngent.wire.event import AnswerCut, AssistantDelta, AssistantText, TurnFinished, TurnStarted, Usage

if TYPE_CHECKING:
    from plyngent.kernel.catalog import ToolCatalog
    from plyngent.kernel.executor import ToolExecutor
    from plyngent.kernel.model import AnyMessage
    from plyngent.kernel.session import TurnCtx
    from plyngent.kernel.trail import AnyRecord
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
        failure: Failure | None = None
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
            self._emit(
                TurnFinished(
                    session=session,
                    turn=ctx.turn_id,
                    reason=reason,
                    ends_at=state.cursor,
                    failure="" if failure is None else str(failure),
                ),
            )
            ctx.end()

    # -- the stages ----------------------------------------------------------------

    async def assemble(self, state: TurnState) -> Request:
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
        """
        ctx = state.ctx
        trail = ctx.host.trail
        messages = await trail.derive(ctx.session.session_id, state.cursor)
        view = self._catalog.view(ctx.session.disclosed)
        asked = ctx.host.routes.resolve(ctx.session.route)
        return Request(
            system=ctx.host.prompt.compose(view),
            messages=await self._seen(ctx, messages),
            tools=view.prompts(),
            route="" if asked is None else asked.name,
            cache_key=ctx.session.cache_key,
        )

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

        What the asking cost is said outward here rather than by the model: a model answers a value, and
        every event on this lane is the kernel's to emit. An answer that stopped short is said the same
        way, because both are facts about the request that just came back and neither is the turn's end.

        What the model *is saying* is said outward while it is still saying it — one event per fragment,
        as they arrive — but the answer this returns is the whole one, because that is what the turn
        writes down (§0.13, E94). A model that cannot stream says one thing and no fragments, so this is
        one loop and not two.
        """
        reply = await state.ctx.round_scope().guard(self._answered(state, request))
        if reply.usage.reported:
            self._emit(
                Usage(
                    session=state.ctx.session.session_id,
                    turn=state.ctx.turn_id,
                    prompt=reply.usage.prompt,
                    completion=reply.usage.completion,
                    cached=reply.usage.cached,
                ),
            )
        if reply.cut is not None:
            self._emit(
                AnswerCut(session=state.ctx.session.session_id, turn=state.ctx.turn_id, reason=str(reply.cut)),
            )
        return reply

    async def _answered(self, state: TurnState, request: Request) -> Reply:
        """Every fragment of the answer said outward as it arrives, and then the answer itself."""
        async for piece in state.ctx.host.model.stream(request):
            if isinstance(piece, Delta):
                if piece.text:
                    self._emit(
                        AssistantDelta(session=state.ctx.session.session_id, turn=state.ctx.turn_id, text=piece.text),
                    )
                continue
            return piece
        msg = "the model's stream ended without the answer it was adding up to"
        raise ModelFailed(Failure.PROTOCOL, msg)

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
        if not (effects.changed or effects.external or effects.approvals):
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
