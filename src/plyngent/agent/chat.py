from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.lmproto.openai_compatible.model import SystemChatMessage, UserChatMessage

from .budget import DEFAULT_CONTEXT_MAX_CHARS, DEFAULT_TOOL_RESULT_MAX_CHARS
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
    system_prompt: str | None
    max_tool_result_chars: int
    parallel_tools: bool
    max_context_chars: int
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
        system_prompt: str | None = None,
        max_tool_result_chars: int = DEFAULT_TOOL_RESULT_MAX_CHARS,
        parallel_tools: bool = True,
        max_context_chars: int = DEFAULT_CONTEXT_MAX_CHARS,
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
        self.max_context_chars = max_context_chars
        self.messages = list(messages) if messages is not None else []
        self.pending_retry_text = None
        self._ensure_system_prompt()

    def _ensure_system_prompt(self) -> None:
        """Prepend system prompt once when configured and history has none."""
        if not self.system_prompt:
            return
        if self.messages and isinstance(self.messages[0], SystemChatMessage):
            return
        self.messages.insert(0, SystemChatMessage(content=self.system_prompt))

    async def load_history(self) -> None:
        """Replace in-memory messages from the bound memory session."""
        if self.memory is None or self.session_id is None:
            msg = "load_history requires memory and session_id"
            raise RuntimeError(msg)
        self.messages = await self.memory.list_messages(self.session_id)
        self.pending_retry_text = None
        self._ensure_system_prompt()

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
                max_tool_result_chars=self.max_tool_result_chars,
                parallel_tools=self.parallel_tools,
                max_context_chars=self.max_context_chars,
            ):
                yield event
            completed = True
        except BaseException:
            if not completed:
                self._rollback_turn(pre_len, user_msg.content)
            raise

        for message in self.messages[pre_len:]:
            await self._persist(message)
        self.pending_retry_text = None

    async def run(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user message, run the tool loop, yield events, persist only on success."""
        self._ensure_system_prompt()
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
        self._ensure_system_prompt()
        user_msg = UserChatMessage(content=text)
        self.messages.append(user_msg)
        async for event in self._run_from_user_message(user_msg):
            yield event
