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
    stream_enabled: bool = True
    verbose: bool = False
    # One-shot / scripts: never prompt to raise tool-loop limits.
    interactive_limits: bool = True
    # When False, skip destructive-tool confirms (e.g. --yes).
    confirm_destructive: bool | None = None
    # Set by /edit; REPL sends as the next user turn then clears.
    pending_user_text: str | None = None
    client: ChatClient = field(init=False)
    agent: ChatAgent = field(init=False)
    session_id: int | None = None

    def __post_init__(self) -> None:
        # DeepSeek client uses a compatible but distinct param type; treat as ChatClient.
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.agent = self._make_agent()
        self.sync_display_flags()

    def sync_display_flags(self) -> None:
        from plyngent.cli.display import set_verbose_tool_results

        set_verbose_tool_results(self.verbose)

    def _workspace_key(self) -> str:
        key = normalize_workspace(self.workspace)
        if key is None:
            msg = "workspace path is required"
            raise RuntimeError(msg)
        return key

    def _confirm_destructive(self) -> bool:
        if self.confirm_destructive is not None:
            return self.confirm_destructive
        return self.config.agent_config.confirm_destructive

    def _tool_registry(self) -> ToolRegistry | None:
        if not self.tools_enabled:
            return None
        from plyngent.cli.limits import prompt_confirm_tool_async
        from plyngent.tools.danger import classify_danger

        if self._confirm_destructive():
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
        on_limit = prompt_continue_limit_async if self.interactive_limits else None
        return ChatAgent(
            self.client,
            model=self.model,
            tools=self._tool_registry(),
            memory=self.memory,
            session_id=self.session_id,
            max_rounds=self.max_rounds,
            on_limit=on_limit,
            stream=self.stream_enabled,
            system_prompt=system_prompt,
            max_tool_result_chars=agent_cfg.max_tool_result_chars,
            parallel_tools=agent_cfg.parallel_tools,
            max_context_tokens=agent_cfg.max_context_tokens,
        )

    def rebuild_client(self) -> None:
        """Recreate client and agent after provider/model/tools change."""
        messages = list(self.agent.messages)
        # Preserve live stream toggle if agent already exists.
        if hasattr(self, "agent"):
            self.stream_enabled = self.agent.stream
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.agent = self._make_agent()
        self.agent.messages = messages
        self.sync_display_flags()

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

    async def persist_llm_selection(self) -> None:
        """Write current provider/model onto the active session row (if any)."""
        if self.session_id is None:
            return
        _ = await self.memory.update_session_llm(
            self.session_id,
            provider_name=self.provider_name,
            model=self.model,
        )

    def _try_set_provider(self, pname: str) -> bool:
        import click

        from plyngent.cli.selection import select_provider
        from plyngent.runtime import ProviderNotSupportedError

        if pname not in self.config.providers:
            return False
        try:
            name, provider = select_provider(
                self.config.providers,
                preferred=pname,
                interactive=False,
            )
        except click.ClickException, ProviderNotSupportedError:
            return False
        if name != self.provider_name or provider is not self.provider:
            self.provider_name = name
            self.provider = provider
            return True
        return False

    def _try_set_model(self, model_id: str) -> bool:
        import click

        from plyngent.cli.selection import select_model

        try:
            resolved = select_model(self.provider, preferred=model_id, interactive=False)
        except click.ClickException:
            return False
        if resolved != self.model:
            self.model = resolved
            return True
        return False

    def apply_session_llm(self, row: SessionRow) -> bool:
        """Apply stored provider/model from ``row`` when still valid in config.

        Returns True when selection changed (caller should rebuild agent).
        """
        changed = False
        pname = row.provider_name
        if pname:
            changed = self._try_set_provider(pname) or changed
            if row.model and self._try_set_model(row.model):
                changed = True
            elif changed and self.provider.models and self.model not in self.provider.models:
                self.model = next(iter(sorted(self.provider.models.keys())))
            return changed
        if row.model:
            return self._try_set_model(row.model)
        return False

    async def new_session(self, name: str = "chat") -> None:
        session = await self.memory.create_session(
            name=name,
            workspace=self.workspace,
            provider_name=self.provider_name,
            model=self.model,
        )
        self.session_id = session.sid
        self.agent = self._make_agent()

    async def rename_current_session(self, name: str) -> SessionRow:
        if self.session_id is None:
            msg = "no active session"
            raise ValueError(msg)
        return await self.memory.rename_session(self.session_id, name)

    async def delete_session_and_maybe_replace(self, sid: int) -> bool:
        """Hard-delete ``sid``. If it was current, start a new empty session.

        Returns True when the deleted session was the active one.
        """
        was_current = self.session_id == sid
        ok = await self.memory.delete_session(sid)
        if not ok:
            msg = f"session not found: {sid}"
            raise ValueError(msg)
        if was_current:
            await self.new_session()
        return was_current

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
        if self.apply_session_llm(row):
            self.rebuild_client()
        else:
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
        if self.apply_session_llm(latest):
            self.rebuild_client()
        else:
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

        # Prefer last API prompt_tokens to drive soft-compact toward real size.
        hint: int | None = None
        sent_est: int | None = None
        if not self.agent.last_request_usage.is_zero():
            from plyngent.agent.budget import estimate_messages_tokens

            hint = self.agent.last_request_usage.prompt_tokens
            # Approximate: calibrate against current full history char-est.
            sent_est = estimate_messages_tokens(messages)
        summary = await summarize_messages(
            self.client,
            messages,
            model=self.model,
            max_context_tokens=self.agent.max_context_tokens,
            prompt_tokens_hint=hint,
            sent_estimate_tokens=sent_est,
            system_prompt=self.config.agent_config.compact_system_prompt or None,
            user_prefix=self.config.agent_config.compact_user_prefix or None,
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
            seed_text=self.config.agent_config.compact_seed_text or None,
        )
        self.agent.messages = list(seed)
        for message in seed:
            _ = await self.memory.append_message(new_id, message)
        return old_id, new_id, summary
