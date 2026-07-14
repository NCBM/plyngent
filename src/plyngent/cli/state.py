from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from plyngent.agent import ChatAgent, ChatClient, ToolRegistry
from plyngent.agent.loop import DEFAULT_MAX_ROUNDS
from plyngent.memory.database.store import normalize_workspace
from plyngent.runtime import create_client
from plyngent.tools import DEFAULT_TOOLS, set_workspace_root

if TYPE_CHECKING:
    from plyngent.config.models import Provider
    from plyngent.config.store import ConfigStore
    from plyngent.memory import MemoryStore
    from plyngent.memory.database.schema import Session as SessionRow


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
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.agent = self._make_agent()

    def _workspace_key(self) -> str:
        key = normalize_workspace(self.workspace)
        if key is None:
            msg = "workspace path is required"
            raise RuntimeError(msg)
        return key

    def _tool_registry(self) -> ToolRegistry | None:
        if not self.tools_enabled:
            return None
        from plyngent.cli.limits import prompt_confirm_tool
        from plyngent.tools.danger import classify_danger

        agent_cfg = self.config.agent_config
        if agent_cfg.confirm_destructive:
            return ToolRegistry(
                list(DEFAULT_TOOLS),
                danger=classify_danger,
                on_confirm=prompt_confirm_tool,
            )
        return ToolRegistry(list(DEFAULT_TOOLS))

    def _make_agent(self) -> ChatAgent:
        from plyngent.cli.limits import prompt_continue_limit

        agent_cfg = self.config.agent_config
        system_prompt = agent_cfg.system_prompt or None
        return ChatAgent(
            self.client,
            model=self.model,
            tools=self._tool_registry(),
            memory=self.memory,
            session_id=self.session_id,
            max_rounds=self.max_rounds,
            on_limit=prompt_continue_limit,
            stream=True,
            system_prompt=system_prompt,
            max_tool_result_chars=agent_cfg.max_tool_result_chars,
            parallel_tools=agent_cfg.parallel_tools,
            max_context_chars=agent_cfg.max_context_chars,
        )

    def rebuild_client(self) -> None:
        """Recreate client and agent after provider/model/tools change."""
        messages = list(self.agent.messages)
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.agent = self._make_agent()
        self.agent.messages = messages

    def _apply_session_workspace(self, row: SessionRow) -> None:
        """Bind tools/REPL workspace to the session's directory when set."""
        if not row.workspace:
            return
        path = Path(row.workspace).expanduser().resolve()
        if not path.is_dir():
            msg = f"session {row.sid} workspace is not a directory: {path}"
            raise ValueError(msg)
        self.workspace = path
        _ = set_workspace_root(path)

    async def new_session(self, name: str = "chat") -> None:
        session = await self.memory.create_session(name=name, workspace=self.workspace)
        self.session_id = session.sid
        self.agent = self._make_agent()
        self.agent.pending_retry_text = None

    async def resume_session(self, session_id: int) -> None:
        row = await self.memory.get_session(session_id)
        if row is None:
            msg = f"session not found: {session_id}"
            raise ValueError(msg)
        expected = self._workspace_key()
        if row.workspace is not None and row.workspace != expected:
            msg = (
                f"session {session_id} is bound to workspace {row.workspace!r}, "
                f"not current {expected!r} (use matching --workspace or /resume from that dir)"
            )
            raise ValueError(msg)
        self._apply_session_workspace(row)
        self.session_id = session_id
        self.agent = self._make_agent()
        await self.agent.load_history()

    async def resume_latest_or_new(self, name: str = "chat") -> str:
        """Resume latest session for this workspace, or create one if none exist."""
        sessions = await self.memory.list_sessions(workspace=self.workspace)
        if not sessions:
            await self.new_session(name=name)
            return "new"
        latest = max(sessions, key=lambda s: (s.updated_at, s.sid))
        await self.resume_session(latest.sid)
        return "resume"
