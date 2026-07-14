from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from plyngent.agent import ChatAgent
from plyngent.cli.retry import retry_pending_with_retries, run_turn_with_retries, sleep_cancellable
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
    UserChatMessage,
)
from plyngent.memory import MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import pytest


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
        if stream:

            async def empty() -> AsyncIterator[ChatCompletionChunk]:
                empty_chunks: list[ChatCompletionChunk] = []
                for chunk in empty_chunks:
                    yield chunk

            return empty()
        return _response(AssistantChatMessage(content="ok"))


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
    assert client.calls == 3  # noqa: PLR2004
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2  # noqa: PLR2004
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hello"
    await store.close()


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
    assert await store.list_messages(session.sid) == []

    # Now succeed on manual /retry path
    client.fail_times = client.calls  # next call succeeds
    ok2 = await retry_pending_with_retries(agent)
    # retry_pending uses default delays; force zero-wait already patched
    # But fail_times set so first attempt of manual retry succeeds immediately
    assert ok2 is True
    assert agent.pending_retry_text is None
    loaded = await store.list_messages(session.sid)
    assert len(loaded) == 2  # noqa: PLR2004
    assert isinstance(loaded[0], UserChatMessage)
    assert loaded[0].content == "hold-me"
    await store.close()
