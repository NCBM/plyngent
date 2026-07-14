from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from plyngent.agent import ChatAgent, ChatClient, ToolRegistry
from plyngent.agent.loop import DEFAULT_MAX_ROUNDS
from plyngent.runtime import create_client
from plyngent.tools import DEFAULT_TOOLS

if TYPE_CHECKING:
    from pathlib import Path

    from plyngent.config.models import Provider
    from plyngent.config.store import ConfigStore
    from plyngent.memory import MemoryStore


@dataclass
class ReplState:
    """Mutable REPL session state."""

    config: ConfigStore
    memory: MemoryStore
    workspace: Path
    provider_name: str
    provider: Provider
    model: str
    tools_enabled: bool
    max_rounds: int = DEFAULT_MAX_ROUNDS
    client: ChatClient = field(init=False)
    agent: ChatAgent = field(init=False)
    session_id: int | None = None

    def __post_init__(self) -> None:
        # DeepSeek client uses a compatible but distinct param type; treat as ChatClient.
        self.client = cast("ChatClient", create_client(self.provider))
        self.agent = self._make_agent()

    def _tool_registry(self) -> ToolRegistry | None:
        if not self.tools_enabled:
            return None
        return ToolRegistry(list(DEFAULT_TOOLS))

    def _make_agent(self) -> ChatAgent:
        from plyngent.cli.limits import prompt_continue_limit

        return ChatAgent(
            self.client,
            model=self.model,
            tools=self._tool_registry(),
            memory=self.memory,
            session_id=self.session_id,
            max_rounds=self.max_rounds,
            on_limit=prompt_continue_limit,
        )

    def rebuild_client(self) -> None:
        """Recreate client and agent after provider/model/tools change."""
        messages = list(self.agent.messages)
        self.client = cast("ChatClient", create_client(self.provider))
        self.agent = self._make_agent()
        self.agent.messages = messages

    async def new_session(self, name: str = "chat") -> None:
        session = await self.memory.create_session(name=name)
        self.session_id = session.sid
        self.agent = self._make_agent()

    async def resume_session(self, session_id: int) -> None:
        row = await self.memory.get_session(session_id)
        if row is None:
            msg = f"session not found: {session_id}"
            raise ValueError(msg)
        self.session_id = session_id
        self.agent = self._make_agent()
        await self.agent.load_history()

    async def resume_latest_or_new(self, name: str = "chat") -> str:
        """Resume the most recently updated session, or create one if none exist."""
        sessions = await self.memory.list_sessions()
        if not sessions:
            await self.new_session(name=name)
            return "new"
        latest = max(sessions, key=lambda s: (s.updated_at, s.sid))
        await self.resume_session(latest.sid)
        return "resume"
