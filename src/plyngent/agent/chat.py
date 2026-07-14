from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import UserChatMessage

from .loop import DEFAULT_MAX_ROUNDS, run_chat_loop

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage
    from plyngent.memory import MemoryStore

    from .client import ChatClient
    from .events import AgentEvent
    from .tools import ToolRegistry


class ChatAgent:
    """Thin wrapper: chat client + optional tools + optional memory bind."""

    client: ChatClient
    model: str
    tools: ToolRegistry | None
    memory: MemoryStore | None
    session_id: int | None
    max_rounds: int
    temperature: float | None
    messages: list[AnyChatMessage]

    def __init__(  # noqa: PLR0913
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
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.memory = memory
        self.session_id = session_id
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.messages = list(messages) if messages is not None else []

    async def load_history(self) -> None:
        """Replace in-memory messages from the bound memory session."""
        if self.memory is None or self.session_id is None:
            msg = "load_history requires memory and session_id"
            raise RuntimeError(msg)
        self.messages = await self.memory.list_messages(self.session_id)

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

    async def run(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user message, run the tool loop, yield events, persist new messages."""
        user_msg = UserChatMessage(content=user_text)
        self.messages.append(user_msg)
        await self._persist(user_msg)

        start_len = len(self.messages)
        async for event in run_chat_loop(
            self.client,
            self.messages,
            model=self.model,
            tools=self.tools,
            max_rounds=self.max_rounds,
            temperature=self.temperature,
        ):
            yield event

        # Persist messages appended by the loop after the user message.
        for message in self.messages[start_len:]:
            await self._persist(message)
