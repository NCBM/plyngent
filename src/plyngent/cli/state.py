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
        from plyngent.cli.limits import prompt_confirm_tool_async
        from plyngent.tools.danger import classify_danger

        agent_cfg = self.config.agent_config
        if agent_cfg.confirm_destructive:
            return ToolRegistry(
                list(DEFAULT_TOOLS),
                danger=classify_danger,
                on_confirm=prompt_confirm_tool_async,
            )
        return ToolRegistry(list(DEFAULT_TOOLS))

    def _make_agent(self) -> ChatAgent:
        from plyngent.cli.limits import prompt_continue_limit_async

        agent_cfg = self.config.agent_config
        system_prompt = agent_cfg.system_prompt or None
        return ChatAgent(
            self.client,
            model=self.model,
            tools=self._tool_registry(),
            memory=self.memory,
            session_id=self.session_id,
            max_rounds=self.max_rounds,
            on_limit=prompt_continue_limit_async,
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

    def _set_workspace(self, path: Path) -> None:
        """Update REPL + tool workspace root."""
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            msg = f"workspace is not a directory: {resolved}"
            raise ValueError(msg)
        self.workspace = resolved
        _ = set_workspace_root(resolved)

    def _apply_session_workspace(self, row: SessionRow) -> None:
        """Bind tools/REPL workspace to the session's directory when set."""
        if not row.workspace:
            return
        self._set_workspace(Path(row.workspace))

    async def new_session(self, name: str = "chat") -> None:
        session = await self.memory.create_session(name=name, workspace=self.workspace)
        self.session_id = session.sid
        self.agent = self._make_agent()
        self.agent.pending_retry_text = None

    async def resume_session(self, session_id: int) -> None:
        """Load a session; on workspace mismatch, prompt keep / rebind / abort."""
        from plyngent.cli.limits import prompt_workspace_mismatch

        row = await self.memory.get_session(session_id)
        if row is None:
            msg = f"session not found: {session_id}"
            raise ValueError(msg)

        current = self._workspace_key()
        if row.workspace is None:
            # Legacy unbound session: attach to the current workspace.
            row = await self.memory.update_session_workspace(session_id, self.workspace)
        elif row.workspace != current:
            choice = prompt_workspace_mismatch(session_id, row.workspace, current)
            if choice == "abort":
                msg = "resume aborted"
                raise ValueError(msg)
            if choice == "rebind":
                row = await self.memory.update_session_workspace(session_id, self.workspace)
            else:
                # keep: switch live workspace to the session binding
                try:
                    self._set_workspace(Path(row.workspace))
                except ValueError as exc:
                    msg = f"cannot keep session workspace: {exc}"
                    raise ValueError(msg) from exc

        self.session_id = session_id
        self.agent = self._make_agent()
        await self.agent.load_history()

    async def resume_latest_or_new(self, name: str = "chat") -> str:
        """Resume most recently updated session for this workspace, or create one."""
        latest = await self.memory.get_latest_session(workspace=self.workspace)
        if latest is None:
            await self.new_session(name=name)
            return "new"
        # Same-workspace list: no mismatch prompt expected.
        self.session_id = latest.sid
        self.agent = self._make_agent()
        await self.agent.load_history()
        _ = await self.memory.touch_session(latest.sid)
        return "resume"

    async def compact_to_new_session(self, *, name: str | None = None) -> tuple[int, int, str]:
        """Soft-compact + model-summarize current history into a new workspace session.

        Returns ``(old_session_id, new_session_id, summary)``.
        """
        from plyngent.agent.compact import build_compacted_seed_messages, summarize_messages

        old_id = self.session_id
        if old_id is None:
            msg = "no active session to compact"
            raise ValueError(msg)
        messages = list(self.agent.messages)
        if len(messages) < 1:
            msg = "nothing to compact (empty history)"
            raise ValueError(msg)

        summary = await summarize_messages(
            self.client,
            messages,
            model=self.model,
            max_context_chars=self.agent.max_context_chars,
        )
        session_name = name or f"compact-from-{old_id}"
        await self.new_session(name=session_name)
        new_id = self.session_id
        if new_id is None:
            msg = "failed to create compact session"
            raise RuntimeError(msg)

        seed = build_compacted_seed_messages(
            summary,
            system_prompt=self.agent.system_prompt,
            source_session_id=old_id,
        )
        self.agent.messages = list(seed)
        for message in seed:
            _ = await self.memory.append_message(new_id, message)
        self.agent.pending_retry_text = None
        return old_id, new_id, summary
