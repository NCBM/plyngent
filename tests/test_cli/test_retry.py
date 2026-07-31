from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import pytest

from plyngent.agent import ChatAgent
from plyngent.cli.retry import (
    DEFAULT_MAX_AUTO_RETRIES,
    DEFAULT_RETRY_DELAYS_SECONDS,
    _wait_for_retry,
    default_retry_delays,
    retry_pending_with_retries,
    run_turn_with_retries,
    sleep_cancellable,
)
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    ChunkChoice,
    DeltaMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_default_retry_delays_schedule() -> None:
    assert DEFAULT_MAX_AUTO_RETRIES == 10
    assert default_retry_delays(0) == ()
    assert default_retry_delays(4) == (5.0, 10.0, 15.0, 20.0)
    delays = default_retry_delays(10)
    assert delays[:4] == (5.0, 10.0, 15.0, 20.0)
    assert delays[4:] == (30.0, 40.0, 50.0, 60.0, 70.0, 80.0)
    assert delays == DEFAULT_RETRY_DELAYS_SECONDS
    assert len(DEFAULT_RETRY_DELAYS_SECONDS) == 10


def _response(message: AssistantChatMessage) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="1",
        object="chat.completion",
        created=0,
        model="test",
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                logprobs={},
                finish_reason="stop",
            )
        ],
        system_fingerprint="",
        usage={},
    )


async def _as_stream(response: ChatCompletionResponse) -> AsyncIterator[ChatCompletionChunk]:
    message = response.choices[0].message
    content = message.content if isinstance(message.content, str) else ""
    yield ChatCompletionChunk(
        id=response.id,
        object="chat.completion.chunk",
        created=response.created,
        model=response.model,
        choices=[
            ChunkChoice(
                index=0,
                delta=DeltaMessage(content=content),
                finish_reason="stop",
            )
        ],
    )


class FlakyClient:
    calls: int
    fail_times: int

    def __init__(self, fail_times: int) -> None:
        self.calls = 0
        self.fail_times = fail_times

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[False] = False
    ) -> ChatCompletionResponse: ...

    @overload
    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: Literal[True]
    ) -> AsyncIterator[ChatCompletionChunk]: ...

    async def chat_completions(
        self, param: ChatCompletionsParam, *, stream: bool = False
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        del param
        self.calls += 1
        if self.calls <= self.fail_times:
            msg = f"fail-{self.calls}"
            raise RuntimeError(msg)
        response = _response(AssistantChatMessage(content="ok"))
        if stream:
            return _as_stream(response)
        return response


async def test_sleep_cancellable_completes() -> None:
    assert await sleep_cancellable(0.01) is True


async def test_auto_retry_eventually_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_seconds: float) -> bool:
        return True

    monkeypatch.setattr("plyngent.cli.retry.sleep_cancellable", no_wait)
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")
    client = FlakyClient(fail_times=2)
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)

    ok = await run_turn_with_retries(
        agent,
        starter=lambda: agent.run("hello"),
        delays=(0.01, 0.01, 0.01),
    )
    assert ok is True
    assert client.calls == 3
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hello"
    await store.close()


async def test_run_cancellable_cancels_task() -> None:
    import asyncio

    from plyngent.cli.retry import run_cancellable

    started = asyncio.Event()

    async def hang() -> None:
        started.set()
        await asyncio.sleep(60)

    task = asyncio.create_task(run_cancellable(hang()))
    _ = await started.wait()
    _ = task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancel_turn_rolls_back_and_sets_pending() -> None:
    import asyncio

    from plyngent.cli.display import render_events
    from plyngent.cli.retry import run_cancellable

    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    class HangClient:
        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param, stream
            await asyncio.sleep(60)
            return _response(AssistantChatMessage(content="never"))

    agent = ChatAgent(HangClient(), model="m", memory=store, session_id=session.sid)
    turn = asyncio.create_task(run_cancellable(render_events(agent.run("cancel-me"))))
    await asyncio.sleep(0.05)
    _ = turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn

    assert agent.pending_retry_text == "cancel-me"
    assert len(agent.messages) == 1
    assert isinstance(agent.messages[0], UserChatMessage)
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 1
    assert isinstance(loaded[0], UserChatMessage)
    await store.close()


async def test_second_ctrl_c_during_cancel_message_stays_in_repl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a second Ctrl+C during the cancel message must not exit.

    After the turn task is cancelled the asyncio SIGINT handler is gone, so the
    second Ctrl+C arrives as a KeyboardInterrupt while the "cancelled" message
    is printing. It must be swallowed and control must return to the REPL
    (``False``), not propagate out of ``run_turn_with_retries``.
    """
    import asyncio

    calls: list[int] = [0]

    def raiser(message: object = None, **kwargs: object) -> None:
        calls[0] += 1
        if calls[0] == 1:
            raise KeyboardInterrupt

    monkeypatch.setattr("plyngent.cli.retry.click.secho", raiser)

    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")

    class HangClient:
        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param, stream
            await asyncio.sleep(60)
            return _response(AssistantChatMessage(content="never"))

    agent = ChatAgent(HangClient(), model="m", memory=store, session_id=session.sid)
    turn = asyncio.create_task(run_turn_with_retries(agent, starter=lambda: agent.run("cancel-me"), delays=()))
    await asyncio.sleep(0.05)
    _ = turn.cancel()  # first Ctrl+C: cancels the in-flight turn
    ok = await turn  # must return False instead of raising KeyboardInterrupt
    assert ok is False
    assert agent.pending_retry_text == "cancel-me"
    await store.close()


async def test_second_ctrl_c_during_retry_cancel_message_stays_benign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a second Ctrl+C during the retry-cancel message is benign."""

    async def interrupted_sleep(_seconds: float) -> bool:
        return False  # first Ctrl+C interrupted the retry wait

    monkeypatch.setattr("plyngent.cli.retry.sleep_cancellable", interrupted_sleep)

    calls: list[int] = [0]

    def raiser(message: object = None, **kwargs: object) -> None:
        calls[0] += 1
        if calls[0] == 2:  # second Ctrl+C lands on the "auto-retry cancelled" line
            raise KeyboardInterrupt

    monkeypatch.setattr("plyngent.cli.retry.click.secho", raiser)

    assert await _wait_for_retry(1, 3, 0.01) is False


async def test_second_ctrl_c_during_retry_announce_stays_benign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Ctrl+C during the "retrying" announcement does not abort the turn."""

    async def ok_sleep(_seconds: float) -> bool:
        return True

    monkeypatch.setattr("plyngent.cli.retry.sleep_cancellable", ok_sleep)

    calls: list[int] = [0]

    def raiser(message: object = None, **kwargs: object) -> None:
        calls[0] += 1
        if calls[0] == 2:  # Ctrl+C lands on the "retrying" line
            raise KeyboardInterrupt

    monkeypatch.setattr("plyngent.cli.retry.click.secho", raiser)

    assert await _wait_for_retry(1, 3, 0.01) is True


async def test_manual_retry_after_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_seconds: float) -> bool:
        return True

    monkeypatch.setattr("plyngent.cli.retry.sleep_cancellable", no_wait)
    store = await MemoryStore.open(DatabaseConfig())
    session = await store.create_session(name="t")
    client = FlakyClient(fail_times=5)
    agent = ChatAgent(client, model="m", memory=store, session_id=session.sid)

    ok = await run_turn_with_retries(
        agent,
        starter=lambda: agent.run("hold-me"),
        delays=(0.01,),  # one auto-retry only → 2 attempts total, both fail
    )
    assert ok is False
    assert agent.pending_retry_text == "hold-me"
    assert len(await store.list_messages(session.sid)) == 1

    # Now succeed on manual /retry path (no second user message)
    client.fail_times = client.calls  # next call succeeds
    ok2 = await retry_pending_with_retries(agent)
    assert ok2 is True
    assert agent.pending_retry_text is None
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hold-me"
    assert sum(1 for m in loaded if isinstance(m, UserChatMessage)) == 1
    await store.close()
