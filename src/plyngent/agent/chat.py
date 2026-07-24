from __future__ import annotations

from typing import TYPE_CHECKING

from msgspec import UNSET

from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionToolCall,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)

from .budget import (
    DEFAULT_CONTEXT_MAX_TOKENS,
    DEFAULT_TOOL_RESULT_MAX_CHARS,
    estimate_messages_tokens,
)
from .events import UsageEvent
from .loop import DEFAULT_MAX_ROUNDS, run_chat_loop
from .todo_nag import (
    DEFAULT_TODO_NAG_STRATEGY,
    inject_todo_nag_for_stack_with_events,
    parse_todo_nag_strategy,
    refresh_synthetic_todo_nags,
)
from .tools import ToolRegistry
from .usage import TokenUsage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

    from plyngent.lmproto.openai_compatible.model import AnyChatMessage
    from plyngent.memory import MemoryStore

    from .client import ChatClient
    from .events import AgentEvent
    from .todo_nag import TodoNagStrategy
    from .todo_stack import TodoStack

    type LimitContinueHook = Callable[[str], bool | Awaitable[bool]]


def incomplete_turn_user_text(messages: Sequence[AnyChatMessage]) -> str | None:
    """User text for an incomplete turn, if history can be retried.

    Incomplete when the last non-system message is a user message (failed before
    any commit), or tool results after a complete tool batch (failed on a later
    model round — keep tools so retry does not re-execute side effects).
    """
    # Skip leading system prompt only.
    start = 0
    if messages and isinstance(messages[0], SystemChatMessage):
        start = 1
    body = list(messages[start:])
    if not body:
        return None
    last = body[-1]
    if isinstance(last, UserChatMessage):
        return last.content
    if isinstance(last, ToolChatMessage):
        # Walk back to the user that opened this turn.
        for msg in reversed(body):
            if isinstance(msg, UserChatMessage):
                return msg.content
        return None
    return None


def committed_prefix_end(messages: Sequence[AnyChatMessage], user_index: int) -> int:
    """Index of the first uncommitted message after a successful tool batch.

    Committed = complete assistant(tool_calls) + matching tool results for that
    batch, possibly repeated. An assistant without finished tools, or a partial
    text-only assistant mid-stream, is not committed.
    """
    end = user_index + 1
    i = user_index + 1
    n = len(messages)
    while i < n:
        msg = messages[i]
        if not isinstance(msg, AssistantChatMessage):
            break
        tool_calls = msg.tool_calls
        if tool_calls is UNSET or not tool_calls:
            # Final text assistant is only committed on successful turn end.
            break
        expected_ids: list[str] = []
        for call in tool_calls:
            if isinstance(call, AssistantFunctionToolCall):
                expected_ids.append(call.id)
            else:
                expected_ids.append(call.id)
        j = i + 1
        got: list[str] = []
        while j < n and isinstance(messages[j], ToolChatMessage):
            tool_msg = messages[j]
            assert isinstance(tool_msg, ToolChatMessage)
            got.append(tool_msg.tool_call_id)
            j += 1
        if len(got) < len(expected_ids):
            # Incomplete tool batch — do not commit this assistant.
            break
        # Allow extra tool messages only if exact prefix match of ids (order may vary).
        if sorted(got[: len(expected_ids)]) != sorted(expected_ids):
            break
        end = j
        i = j
    return end


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
    todo_stack: TodoStack | None
    todo_nag_strategy: TodoNagStrategy
    messages: list[AnyChatMessage]
    session_usage: TokenUsage
    last_turn_usage: TokenUsage
    last_request_usage: TokenUsage
    last_turn_rounds: int
    # Index into messages of the first unpersisted message (checkpoint cursor).
    _persist_from: int

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
        todo_stack: TodoStack | None = None,
        todo_nag_strategy: str | TodoNagStrategy = DEFAULT_TODO_NAG_STRATEGY,
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
        self.todo_stack = todo_stack
        self.todo_nag_strategy = parse_todo_nag_strategy(str(todo_nag_strategy))
        self.messages = list(messages) if messages is not None else []
        self.session_usage = TokenUsage()
        self.last_turn_usage = TokenUsage()
        self.last_request_usage = TokenUsage()
        self.last_turn_rounds = 0
        self._persist_from = len(self.messages)
        self._ensure_system_prompt()

    @property
    def pending_retry_text(self) -> str | None:
        """User text of an incomplete turn that can be continued with :meth:`retry`."""
        return incomplete_turn_user_text(self.messages)

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
        # Prepended system is local-only; shift the checkpoint so indices that
        # already pointed past stored messages stay correct after insert.
        self._persist_from = min(len(self.messages), self._persist_from + 1)

    def replace_messages(
        self,
        messages: Sequence[AnyChatMessage],
        *,
        persisted: bool = True,
        persist_from: int | None = None,
    ) -> None:
        """Replace in-memory history and align the persistence checkpoint.

        When *persist_from* is set, it is used as the checkpoint cursor
        (clamped to ``[0, len(messages)]``). Otherwise *persisted* true means
        all messages are already stored; false means none are.
        """
        self.messages = list(messages)
        if persist_from is not None:
            self._persist_from = max(0, min(persist_from, len(self.messages)))
        else:
            self._persist_from = len(self.messages) if persisted else 0
        self._ensure_system_prompt()

    @property
    def persist_from(self) -> int:
        """Index of the first unpersisted message (checkpoint cursor)."""
        return self._persist_from

    async def load_history(self) -> None:
        """Replace in-memory messages from the bound memory session."""
        if self.memory is None or self.session_id is None:
            msg = "load_history requires memory and session_id"
            raise RuntimeError(msg)
        loaded = await self.memory.list_messages(self.session_id)
        self.replace_messages(loaded, persisted=True)

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

    async def _persist_range(self, start: int, end: int) -> None:
        """Persist messages[start:end] and advance the checkpoint cursor."""
        for message in self.messages[start:end]:
            await self._persist(message)
        self._persist_from = max(self._persist_from, end)

    def _user_index(self, user_msg: UserChatMessage) -> int:
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i] is user_msg:
                return i
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if isinstance(msg, UserChatMessage) and msg.content == user_msg.content:
                return i
        msg = "user message not found in history"
        raise RuntimeError(msg)

    def _find_turn_user_index(self) -> int:
        """Index of the user message that opens the current incomplete turn."""
        text = incomplete_turn_user_text(self.messages)
        if text is None:
            msg = "nothing to retry"
            raise RuntimeError(msg)
        for i in range(len(self.messages) - 1, -1, -1):
            msg = self.messages[i]
            if isinstance(msg, UserChatMessage) and msg.content == text:
                return i
        msg = "nothing to retry"
        raise RuntimeError(msg)

    def _rollback_uncommitted(self, user_index: int) -> None:
        """Drop incomplete suffix; keep committed tool rounds after the user."""
        end = committed_prefix_end(self.messages, user_index)
        del self.messages[end:]

    async def _run_from_user_message(self, user_msg: UserChatMessage) -> AsyncIterator[AgentEvent]:
        """Run the tool loop for an already-appended user message.

        After each completed tool batch, commits assistant+tool messages to the
        DB and keeps them on failure so :meth:`retry` continues without redoing
        side-effecting tools. Unfinished assistant/stream suffix is rolled back.
        """
        user_index = self._user_index(user_msg)
        if self.todo_stack is not None:
            self.todo_stack.begin_turn()
            # Keep forged synthetic_tool nags aligned with the live stack so a
            # previously dirty stack does not re-surface after it was cleaned.
            _ = refresh_synthetic_todo_nags(self.messages, self.todo_stack)

        completed = False
        turn_usage = TokenUsage()
        turn_rounds = 0
        last_request = TokenUsage()
        try:
            # Turn-start nag before first completion; yield events so CLI flushes
            # (synthetic_tool → ToolCall/Result chrome, not glued to text).
            if self.todo_stack is not None and not self.todo_stack.is_empty():
                _injected, nag_events = inject_todo_nag_for_stack_with_events(
                    self.messages,
                    self.todo_stack,
                    kind="turn_start",
                    strategy=self.todo_nag_strategy,
                )
                for nag_event in nag_events:
                    yield nag_event

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
                todo_stack=self.todo_stack,
                todo_nag_strategy=self.todo_nag_strategy,
            ):
                if isinstance(event, UsageEvent):
                    turn_rounds += 1
                    last_request = event.usage
                    turn_usage = turn_usage.add(event.usage)
                    self.session_usage = self.session_usage.add(event.usage)
                # After a full tool batch, commit prefix so failures do not
                # discard work that already had external effects.
                commit_end = committed_prefix_end(self.messages, user_index)
                if commit_end > self._persist_from:
                    await self._persist_range(self._persist_from, commit_end)
                yield event
            completed = True
        except BaseException:
            if not completed:
                self._rollback_uncommitted(user_index)
            raise

        self.last_turn_usage = turn_usage
        self.last_request_usage = last_request
        self.last_turn_rounds = turn_rounds
        if self._persist_from < len(self.messages):
            await self._persist_range(self._persist_from, len(self.messages))

    async def run(self, user_text: str) -> AsyncIterator[AgentEvent]:
        """Append a user message (persist immediately), run the tool loop, yield events."""
        self._ensure_system_prompt()
        user_msg = UserChatMessage(content=user_text)
        self.messages.append(user_msg)
        await self._persist(user_msg)
        self._persist_from = len(self.messages)
        async for event in self._run_from_user_message(user_msg):
            yield event

    def _resolve_aside_tools(
        self,
        *,
        tools: ToolRegistry | bool | None,
        instance_state: object | None,
        session_state: object | None,
    ) -> ToolRegistry | None:
        """Resolve the tools argument for :meth:`run_aside`."""
        if tools is False or tools is None:
            return None
        if tools is True:
            if self.tools is None:
                return None
            return self.tools.clone(
                instance_state=instance_state,
                session_state=session_state,
            )
        reg = tools
        if instance_state is not None:
            reg.set_instance_state(instance_state)
        if session_state is not None:
            reg.set_session_state(session_state)
        return reg

    async def run_aside(
        self,
        user_text: str,
        *,
        include_history: bool = True,
        tools: ToolRegistry | bool | None = False,
        max_rounds: int | None = None,
        system_prompt: str | None = None,
        instance_state: object | None = None,
        session_state: object | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run a side turn that does not mutate this agent or its memory.

        - Message list is forked (optional history copy); never written back.
        - ``memory`` / ``session_id`` are unset on the side agent (no DB).
        - ``todo_stack`` is unset (no nags / main stack thrash).
        - Tools default **off**. ``tools=True`` clones this agent's registry with
          a **fresh session** bag (unless *session_state* is passed) and the
          given *instance_state* (CLI typically shares the host instance for
          workspace identity).
        """
        text = user_text.strip()
        if not text:
            msg = "aside question must not be empty"
            raise ValueError(msg)

        # Prefer explicit session fork; default empty SessionState when tools on.
        aside_session = session_state
        aside_instance = instance_state
        if tools is True or isinstance(tools, ToolRegistry):
            if aside_session is None:
                from plyngent.tools.context import SessionState

                aside_session = SessionState()
            if aside_instance is None and self.tools is not None:
                # Inherit host instance when the main registry holds one.
                aside_instance = getattr(self.tools, "_instance", None)

        aside_tools = self._resolve_aside_tools(
            tools=tools,
            instance_state=aside_instance,
            session_state=aside_session,
        )
        history = list(self.messages) if include_history else []
        rounds = self.max_rounds if max_rounds is None else max_rounds
        if aside_tools is None and max_rounds is None:
            rounds = 1
        aside = ChatAgent(
            self.client,
            model=self.model,
            tools=aside_tools,
            memory=None,
            session_id=None,
            max_rounds=rounds,
            temperature=self.temperature,
            on_limit=self.on_limit,
            stream=self.stream,
            system_prompt=self.system_prompt if system_prompt is None else system_prompt,
            max_tool_result_chars=self.max_tool_result_chars,
            parallel_tools=self.parallel_tools,
            max_context_tokens=self.max_context_tokens,
            todo_stack=None,
            todo_nag_strategy="none",
            messages=history,
        )
        async for event in aside.run(text):
            yield event

    async def retry(self) -> AsyncIterator[AgentEvent]:
        """Continue an incomplete turn without re-appending the user message.

        Supports:
        - History ending with the user (failed before any tool commit / orphan).
        - History ending with tool results (failed after tools; continue model).
        """
        if incomplete_turn_user_text(self.messages) is None:
            msg = "nothing to retry"
            raise RuntimeError(msg)
        user_index = self._find_turn_user_index()
        user_msg = self.messages[user_index]
        assert isinstance(user_msg, UserChatMessage)
        self._ensure_system_prompt()
        # Already-committed messages stay; only new rounds are persisted.
        self._persist_from = len(self.messages)
        async for event in self._run_from_user_message(user_msg):
            yield event
