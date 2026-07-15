from __future__ import annotations

import asyncio
import contextlib
import signal
import time
from typing import TYPE_CHECKING

import click

from plyngent.cli.display import render_events
from plyngent.cli.interrupt import allow_task_cancel, set_sigint_reinstall

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine

    from plyngent.agent import AgentEvent
    from plyngent.agent.chat import ChatAgent

# Wait before retry attempt 1, 2, and 3 (after the first failure).
DEFAULT_RETRY_DELAYS_SECONDS: tuple[float, ...] = (10.0, 20.0, 30.0)
_PREVIEW_LEN = 80


async def sleep_cancellable(seconds: float) -> bool:
    """Sleep in short steps so Ctrl+C can cancel. Returns False if interrupted."""
    deadline = time.monotonic() + seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            await asyncio.sleep(min(0.5, remaining))
    except asyncio.CancelledError:
        return False
    except KeyboardInterrupt:
        return False


async def run_cancellable[T](coro: Coroutine[object, object, T]) -> T:
    """Await ``coro`` as a task; Ctrl+C / SIGINT cancels the task.

    Interactive prompts (max-rounds / destructive confirm) temporarily disable
    task cancel so the user can answer instead of aborting the whole turn.

    Raises:
        asyncio.CancelledError: If the task was cancelled (including via SIGINT).
    """
    task: asyncio.Task[T] = asyncio.create_task(coro)
    loop = asyncio.get_running_loop()
    installed = False

    def _on_sigint() -> None:
        if allow_task_cancel() and not task.done():
            _ = task.cancel()

    def _reinstall() -> None:
        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
        except (NotImplementedError, RuntimeError, ValueError):
            return

    try:
        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        installed = True
        set_sigint_reinstall(_reinstall)
    except (NotImplementedError, RuntimeError, ValueError):
        installed = False

    try:
        return await task
    except KeyboardInterrupt:
        if not task.done():
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        raise asyncio.CancelledError from None
    finally:
        set_sigint_reinstall(None)
        if installed:
            with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                _ = loop.remove_signal_handler(signal.SIGINT)
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


async def _wait_for_retry(attempt: int, max_retries: int, wait: float) -> bool:
    click.secho(
        f"auto-retry {attempt}/{max_retries} in {wait:g}s "
        f"(Ctrl+C to cancel; then /retry later)",
        fg="yellow",
    )
    try:
        ok = await sleep_cancellable(wait)
    except KeyboardInterrupt:
        ok = False
    if not ok:
        click.secho("auto-retry cancelled; use /retry to try again", fg="yellow")
        click.echo()
        return False
    click.secho(f"retrying ({attempt}/{max_retries})…", fg="yellow")
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
    """
    max_retries = len(delays)
    attempt = 0
    current: Callable[[], AsyncIterator[AgentEvent]] = starter
    while True:
        try:
            await run_cancellable(render_events(current()))
        except asyncio.CancelledError:
            # Do not auto-retry cancelled turns — user intent was stop, not retry.
            click.echo()
            click.secho(
                "cancelled; user message kept — use /retry to try again",
                fg="yellow",
            )
            click.echo()
            return False
        except KeyboardInterrupt:
            click.echo()
            click.secho("interrupted", fg="yellow")
            click.echo()
            return False
        except Exception as exc:  # noqa: BLE001 — surface and optionally retry
            click.secho(f"error: {exc}", fg="red")
            if agent.pending_retry_text is not None:
                current = agent.retry
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


async def run_user_text_with_retries(agent: ChatAgent, text: str) -> bool:
    """Send a new user message with auto-retry."""
    return await run_turn_with_retries(agent, starter=lambda: agent.run(text))


async def retry_pending_with_retries(agent: ChatAgent) -> bool:
    """Retry the last failed user turn with auto-retry."""
    if agent.pending_retry_text is None:
        click.echo("nothing to retry")
        return False
    preview = agent.pending_retry_text
    if len(preview) > _PREVIEW_LEN:
        preview = preview[:_PREVIEW_LEN] + "…"
    click.echo(f"retrying: {preview}")
    return await run_turn_with_retries(agent, starter=agent.retry)
