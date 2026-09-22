from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import click

from plyngent.agent import AssistantMessageEvent
from plyngent.cli.display import render_events
from plyngent.cli.interrupt import install_sigint_router, sigint_cancels
from plyngent.cli.limits import reset_auto_continue_turn

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from plyngent.agent import AgentEvent
    from plyngent.agent.chat import ChatAgent

# Auto-retry budget after the first failure (10 attempts by default).
DEFAULT_MAX_AUTO_RETRIES = 10
# First four waits; each further wait is previous + 10s.
_RETRY_BASE_DELAYS_SECONDS: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0)


def default_retry_delays(max_retries: int = DEFAULT_MAX_AUTO_RETRIES) -> tuple[float, ...]:
    """Build wait times: 5, 10, 15, 20, then +10s each step, length *max_retries*."""
    if max_retries <= 0:
        return ()
    delays: list[float] = list(_RETRY_BASE_DELAYS_SECONDS)
    while len(delays) < max_retries:
        delays.append(delays[-1] + 10.0)
    return tuple(delays[:max_retries])


DEFAULT_RETRY_DELAYS_SECONDS: tuple[float, ...] = default_retry_delays()
_PREVIEW_LEN = 80


async def sleep_cancellable(seconds: float) -> bool:
    """Sleep in short steps so Ctrl+C can cancel. Returns False if interrupted."""
    deadline = time.monotonic() + seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            sleeper = asyncio.ensure_future(asyncio.sleep(min(0.5, remaining)))
            with sigint_cancels(sleeper):
                await sleeper
    except asyncio.CancelledError:
        return False
    except KeyboardInterrupt:
        return False


async def run_cancellable[T](coro: Coroutine[object, object, T]) -> T:
    """Await ``coro`` as a task; Ctrl+C / SIGINT cancels the task.

    Interactive prompts (max-rounds / destructive confirm) register their own
    SIGINT target for as long as they wait, so the turn is not cancelled while
    the human is answering (see :func:`~plyngent.cli.interrupt.off_loop_prompt`).

    Raises:
        asyncio.CancelledError: If the task was cancelled (including via SIGINT).
    """
    task: asyncio.Task[T] = asyncio.create_task(coro)
    # Keep a loop-level SIGINT handler in place for the rest of the session, so
    # a Ctrl+C that lands while the loop is parked in its selector cancels this
    # task instead of being raised in the loop's own frame.
    _ = install_sigint_router()
    try:
        with sigint_cancels(task):
            try:
                return await task
            except KeyboardInterrupt:
                # Platforms without asyncio signal handlers (Windows proactor)
                # keep the CLI's KeyboardInterrupt handler, so the interrupt
                # surfaces here while the task is still awaiting.
                if not task.done():
                    _ = task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                raise asyncio.CancelledError from None
    finally:
        # Only cancel if still running (e.g. KeyboardInterrupt path above).
        # Do not cancel a finished task — that would mask success.
        if not task.done():
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _echo_turn_usage(agent: ChatAgent) -> None:
    if agent.last_turn_usage.is_zero() and agent.last_request_usage.is_zero():
        return
    rounds = agent.last_turn_rounds
    parts: list[str] = []
    if not agent.last_request_usage.is_zero():
        # prompt_tokens on the last call ≈ context the model just saw
        req = agent.last_request_usage
        parts.append(
            f"context={req.prompt_tokens} "
            f"(prompt+completion={req.prompt_tokens}+{req.completion_tokens}"
            f"={req.total_tokens}"
            f"{' est' if req.source == 'estimate' else ''})"
        )
    if not agent.last_turn_usage.is_zero() and (
        rounds > 1 or agent.last_turn_usage.total_tokens != agent.last_request_usage.total_tokens
    ):
        label = agent.last_turn_usage.format_line(billed=True)
        if rounds > 1:
            parts.append(f"turn {label} over {rounds} rounds")
        else:
            parts.append(f"turn {label}")
    click.secho(f"[{'; '.join(parts)}]", fg="bright_black")


def _echo_cancel_lines(*lines: str) -> None:
    """Print post-cancel lines, treating a further Ctrl+C as benign.

    The session SIGINT router keeps the loop safe between turns, but platforms
    without asyncio signal handlers (Windows proactor) deliver a second Ctrl+C
    here as a KeyboardInterrupt, which must not exit the REPL instead of
    returning to the prompt.
    """
    try:
        for line in lines:
            if line:
                click.secho(line, fg="yellow")
            else:
                click.echo()
    except KeyboardInterrupt:
        pass


async def _wait_for_retry(attempt: int, max_retries: int, wait: float) -> bool:
    try:
        click.secho(
            f"auto-retry {attempt}/{max_retries} in {wait:g}s (Ctrl+C to cancel; then /retry later)",
            fg="yellow",
        )
        ok = await sleep_cancellable(wait)
    except KeyboardInterrupt:
        ok = False
    if not ok:
        _echo_cancel_lines("auto-retry cancelled; use /retry to try again", "")
        return False
    _echo_cancel_lines(f"retrying ({attempt}/{max_retries})…")
    return True


async def run_turn_with_retries(
    agent: ChatAgent,
    *,
    starter: Callable[[], AsyncIterator[AgentEvent]],
    delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
) -> bool:
    """Run a chat turn with automatic retries on failure.

    ``starter`` produces the event stream for the first attempt (usually
    ``agent.run``). After a failure (history ends with the user message),
    further attempts use ``agent.retry`` so the user message is not duplicated.

    Ctrl+C cancels the in-flight task; user message stays in DB for ``/retry``.

    The retry budget counts *consecutive* failures: an attempt that completed a
    model round proves the connection answered, so the next failure starts a
    fresh budget (and the short first delay) again.

    The ``yyy`` limit auto-continue is turn-scoped: cleared before and after the
    turn, so a retry of the same turn keeps it but the next user turn prompts.
    """
    reset_auto_continue_turn()
    try:
        return await _run_turn_with_retries(agent, starter=starter, delays=delays)
    finally:
        reset_auto_continue_turn()


async def _run_turn_with_retries(  # noqa: C901 — attempt / cancel / error state machine
    agent: ChatAgent,
    *,
    starter: Callable[[], AsyncIterator[AgentEvent]],
    delays: tuple[float, ...],
) -> bool:
    max_retries = len(delays)
    attempt = 0
    recovered = False
    current: Callable[[], AsyncIterator[AgentEvent]] = starter

    async def note_completed_round(events: AsyncIterator[AgentEvent]) -> AsyncIterator[AgentEvent]:
        """Yield *events*, recording that a full model round came back."""
        nonlocal recovered
        async for event in events:
            if isinstance(event, AssistantMessageEvent):
                recovered = True
            yield event

    while True:
        recovered = False
        try:
            await run_cancellable(render_events(note_completed_round(current())))
        except asyncio.CancelledError:
            # Do not auto-retry cancelled turns — user intent was stop, not retry.
            _echo_cancel_lines("", "cancelled; user message kept — use /retry to try again", "")
            return False
        except KeyboardInterrupt:
            _echo_cancel_lines("", "interrupted", "")
            return False
        except Exception as exc:  # noqa: BLE001 — surface and optionally retry
            click.secho(f"error: {exc}", fg="red")
            if agent.pending_retry_text is not None:
                current = agent.retry
            if recovered:
                # The connection answered a round before failing again, so this
                # is a fresh outage: restart the budget and the delay schedule.
                attempt = 0
            if attempt >= max_retries:
                if agent.pending_retry_text is not None:
                    click.secho(
                        "auto-retry exhausted; use /retry to try again, or send a new message",
                        fg="yellow",
                    )
                click.echo()
                return False
            wait = delays[attempt]
            attempt += 1
            if not await _wait_for_retry(attempt, max_retries, wait):
                return False
        else:
            _echo_turn_usage(agent)
            return True


async def run_user_text_with_retries(
    agent: ChatAgent,
    text: str,
    *,
    delays: tuple[float, ...] = DEFAULT_RETRY_DELAYS_SECONDS,
) -> bool:
    """Send a new user message with auto-retry (empty ``delays`` = no auto-retry)."""
    return await run_turn_with_retries(agent, starter=lambda: agent.run(text), delays=delays)


async def retry_pending_with_retries(agent: ChatAgent) -> bool:
    """Retry/continue an incomplete turn with auto-retry.

    Continues from committed tool results when present (does not re-run them).
    """
    if agent.pending_retry_text is None:
        click.echo("nothing to retry")
        return False
    preview = agent.pending_retry_text
    if len(preview) > _PREVIEW_LEN:
        preview = preview[:_PREVIEW_LEN] + "…"
    from plyngent.lmproto.openai_compatible.model import ToolChatMessage

    if agent.messages and isinstance(agent.messages[-1], ToolChatMessage):
        click.echo(f"continuing after tools (user: {preview})")
    else:
        click.echo(f"retrying: {preview}")
    return await run_turn_with_retries(agent, starter=agent.retry)
