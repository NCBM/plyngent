from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import UserChatMessage

from .loop import DEFAULT_MAX_ROUNDS, run_chat_loop

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage
    from plyngent.memory import MemoryStore

    from .client import ChatClient
    from .events import AgentEvent
    from .tools import ToolRegistry

    type LimitContinueHook = Callable[[str], bool]


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
    messages: list[AnyChatMessage]
    pending_retry_text: str | None

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
        self.messages = list(messages) if messages is not None else []
        self.pending_retry_text = None

    async def load_history(self) -> None:
        """Replace in-memory messages from the bound memory session."""
        if self.memory is None or self.session_id is None:
            msg = "load_history requires memory and session_id"
            raise RuntimeError(msg)
        self.messages = await self.memory.list_messages(self.session_id)
        self.pending_retry_text = None

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

    def _rollback_turn(self, pre_len: int, user_text: str) -> None:
        del self.messages[pre_len:]
        self.pending_retry_text = user_text

    async def _run_from_user_message(self, user_msg: UserChatMessage) -> AsyncIterator[AgentEvent]:
        """Run the tool loop for an already-appended user message; persist only on success."""
        pre_len = len(self.messages) - 1
        if pre_len < 0 or self.messages[pre_len] is not user_msg:
            pre_len = len(self.messages)
            self.messages.append(user_msg)

        completed = False
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
            ):
                yield event
            completed = True
        except BaseException:
            # Includes CancelledError / KeyboardInterrupt paths via task cancel.
            if not completed:
                self._rollback_turn(pre_len, user_msg.content)
            raise

        for message in self.messages[pre_len:]:
            await self._persist(message)
        self.pending_retry_text = None

    async def run(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user message, run the tool loop, yield events, persist only on success."""
        user_msg = UserChatMessage(content=user_text)
        self.messages.append(user_msg)
        async for event in self._run_from_user_message(user_msg):
            yield event

    async def retry(self) -> AsyncIterator[AgentEvent]:
        """Re-run the last failed user turn (no duplicate user message in history/DB)."""
        text = self.pending_retry_text
        if text is None:
            msg = "nothing to retry"
            raise RuntimeError(msg)
        user_msg = UserChatMessage(content=text)
        self.messages.append(user_msg)
        async for event in self._run_from_user_message(user_msg):
            yield event
