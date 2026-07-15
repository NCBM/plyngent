from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import SystemChatMessage, UserChatMessage

from .budget import (
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    estimate_messages_tokens,
)
from .events import UsageEvent
from .loop import DEFAULT_MAX_ROUNDS, run_chat_loop
from .usage import TokenUsage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage
    from plyngent.memory import MemoryStore

    from .client import ChatClient
    from .events import AgentEvent
    from .tools import ToolRegistry

    type LimitContinueHook = Callable[[str], bool | Awaitable[bool]]


class ChatAgent:
    """Thin wrapper: chat client + optional tools + optional memory bind."""

    client: ChatClient
    model: str
    tools: ToolRegistry | None
    memory: MemoryStore | None
    session_id: int | None
    max_rounds: int
    temperature: float | None
    on_limit: LimitContinueHook | None
    stream: bool
    system_prompt: str | None
    max_tool_result_chars: int
    parallel_tools: bool
    max_context_tokens: int
    messages: list[AnyChatMessage]
    pending_retry_text: str | None
    session_usage: TokenUsage
    last_turn_usage: TokenUsage
    last_request_usage: TokenUsage
    last_turn_rounds: int

    def __init__(
        self,
        client: ChatClient,
        *,
        model: str,
        tools: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        session_id: int | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        temperature: float | None = None,
        messages: Sequence[AnyChatMessage] | None = None,
        on_limit: LimitContinueHook | None = None,
        stream: bool = True,
        system_prompt: str | None = None,
        max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
        parallel_tools: bool = True,
        max_context_tokens: int = DEFAULT_CONTEXT_MAX_TOKENS,
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.on_limit = on_limit
        self.stream = stream
        self.system_prompt = system_prompt
        self.max_tool_result_chars = max_tool_result_chars
        self.parallel_tools = parallel_tools
        self.max_context_tokens = max_context_tokens
        self.messages = list(messages) if messages is not None else []
        self.pending_retry_text = None
        self.session_usage = TokenUsage()
        self.last_turn_usage = TokenUsage()
        self.last_request_usage = TokenUsage()
        self.last_turn_rounds = 0
        self._ensure_system_prompt()
        self._sync_pending_from_orphan_user()

    @property
    def context_tokens(self) -> int:
        """Best current context size (tokens).

        Prefers the last model call's ``prompt_tokens`` (API or per-request
        estimate) — that is the real size of the context the model just saw.
        Before any call, falls back to a char-based estimate of ``messages``.
        """
        if not self.last_request_usage.is_zero():
            return self.last_request_usage.prompt_tokens
        return estimate_messages_tokens(self.messages)

    @property
    def context_tokens_source(self) -> str:
        """``api`` / ``estimate`` for :attr:`context_tokens`."""
        if not self.last_request_usage.is_zero():
            return self.last_request_usage.source
        return "estimate"

    def _ensure_system_prompt(self) -> None:
        """Prepend system prompt once when configured and history has none."""
        if not self.system_prompt:
            return
        if self.messages and isinstance(self.messages[0], SystemChatMessage):
            return
        self.messages.insert(0, SystemChatMessage(content=self.system_prompt))

    def _sync_pending_from_orphan_user(self) -> None:
        """If history ends with a user message, that turn is incomplete → retryable."""
        if self.messages and isinstance(self.messages[-1], UserChatMessage):
            self.pending_retry_text = self.messages[-1].content
        else:
            self.pending_retry_text = None

    async def load_history(self) -> None:
        """Replace in-memory messages from the bound memory session."""
        if self.memory is None or self.session_id is None:
            msg = "load_history requires memory and session_id"
            raise RuntimeError(msg)
        self.messages = await self.memory.list_messages(self.session_id)
        self._ensure_system_prompt()
        self._sync_pending_from_orphan_user()

    async def bind_session(self, session_id: int, *, load: bool = True) -> None:
        """Attach a memory session id; optionally load existing messages."""
        if self.memory is None:
            msg = "bind_session requires a MemoryStore"
            raise RuntimeError(msg)
        self.session_id = session_id
        if load:
            await self.load_history()

    async def _persist(self, message: AnyChatMessage) -> None:
        if self.memory is not None and self.session_id is not None:
            _ = await self.memory.append_message(self.session_id, message)

    def _user_index(self, user_msg: UserChatMessage) -> int:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i] is user_msg:
                return i
        # Fallback: last matching content user message
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if isinstance(msg, UserChatMessage) and msg.content == user_msg.content:
                return i
        msg = "user message not found in history"
        raise RuntimeError(msg)

    def _rollback_partial(self, user_index: int, user_text: str) -> None:
        """Drop assistant/tool messages after the user; keep user for retry/DB."""
        del self.messages[user_index + 1 :]
        self.pending_retry_text = user_text

    async def _run_from_user_message(self, user_msg: UserChatMessage) -> AsyncIterator[AgentEvent]:
        """Run the tool loop for an already-appended user message.

        On success, persists assistant/tool messages produced after the user.
        On failure, keeps the user message (already in memory/DB) and sets
        ``pending_retry_text`` so ``retry()`` can re-run without duplicating it.
        """
        user_index = self._user_index(user_msg)

        completed = False
        turn_usage = TokenUsage()
        turn_rounds = 0
        last_request = TokenUsage()
        try:
            async for event in run_chat_loop(
                self.client,
                self.messages,
                model=self.model,
                tools=self.tools,
                max_rounds=self.max_rounds,
                temperature=self.temperature,
                on_limit=self.on_limit,
                stream=self.stream,
                max_tool_result_chars=self.max_tool_result_chars,
                parallel_tools=self.parallel_tools,
                max_context_tokens=self.max_context_tokens,
            ):
                if isinstance(event, UsageEvent):
                    # Each tool-loop round re-sends history; sum is billing, not context size.
                    turn_rounds += 1
                    last_request = event.usage
                    turn_usage = turn_usage.add(event.usage)
                    self.session_usage = self.session_usage.add(event.usage)
                yield event
            completed = True
        except BaseException:
            if not completed:
                self._rollback_partial(user_index, user_msg.content)
            raise

        self.last_turn_usage = turn_usage
        self.last_request_usage = last_request
        self.last_turn_rounds = turn_rounds
        for message in self.messages[user_index + 1 :]:
            await self._persist(message)
        self.pending_retry_text = None

    async def run(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user message (persist immediately), run the tool loop, yield events."""
        self._ensure_system_prompt()
        user_msg = UserChatMessage(content=user_text)
        self.messages.append(user_msg)
        await self._persist(user_msg)
        async for event in self._run_from_user_message(user_msg):
            yield event

    async def retry(self) -> AsyncIterator[AgentEvent]:
        """Re-run the incomplete last user turn without appending a new user message.

        Useful after a failed/cancelled turn (user already in memory/DB) or after
        ``load_history`` when the session ends with an orphan user message.
        """
        text = self.pending_retry_text
        if text is None:
            self._sync_pending_from_orphan_user()
            text = self.pending_retry_text
        if text is None:
            msg = "nothing to retry"
            raise RuntimeError(msg)

        self._ensure_system_prompt()
        if self.messages and isinstance(self.messages[-1], UserChatMessage):
            user_msg = self.messages[-1]
            if user_msg.content != text:
                # Pending text out of sync: replace trailing user
                user_msg = UserChatMessage(content=text)
                self.messages[-1] = user_msg
        else:
            user_msg = UserChatMessage(content=text)
            self.messages.append(user_msg)
            # Already persisted on the original run when memory is bound; only
            # persist if this is a reconstructed orphan without DB (no memory).
            if self.memory is None or self.session_id is None:
                await self._persist(user_msg)

        async for event in self._run_from_user_message(user_msg):
            yield event
