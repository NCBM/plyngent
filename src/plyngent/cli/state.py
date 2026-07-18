from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from plyngent.agent import ChatAgent, ChatClient, ToolRegistry
from plyngent.agent.loop import DEFAULT_MAX_ROUNDS
from plyngent.agent.todo_stack import TodoStack
from plyngent.cli.models_source import (
    DEFAULT_MODELS_CACHE_TTL,
    client_supports_models,
    config_model_ids,
    fetch_remote_model_ids,
    model_choices_for_provider,
)
from plyngent.memory.database.store import normalize_workspace
from plyngent.runtime import create_client
from plyngent.tools import DEFAULT_TOOLS, set_todo_stack, set_workspace_root

if TYPE_CHECKING:
    from collections.abc import Sequence

    from plyngent.config.models import Provider
    from plyngent.config.store import ConfigStore
    from plyngent.memory import MemoryStore
    from plyngent.memory.database.schema import Session as SessionRow

type YoloMode = Literal["off", "on", "once"]


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
    # End-of-turn Rich markdown for assistant text (TTY only).
    markdown_enabled: bool = True
    # One-shot / scripts: never prompt to raise tool-loop limits.
    interactive_limits: bool = True
    # Soft destructive-tool confirms: None → derive from config.confirm_destructive.
    # off = confirm; on = skip (sticky); once = skip next user turn then off.
    yolo: YoloMode | None = None
    # Set by /edit; REPL sends as the next user turn then clears.
    pending_user_text: str | None = None
    client: ChatClient = field(init=False)
    agent: ChatAgent = field(init=False)
    session_id: int | None = None
    todo_stack: TodoStack = field(default_factory=TodoStack)
    _todo_persist_tasks: set[object] = field(default_factory=set, init=False, repr=False)
    # Session ids for Tab complete (updated when listing/creating/resuming).
    _session_id_cache: list[int] = field(default_factory=list, init=False, repr=False)
    # Remote GET /models cache (per provider base).
    _remote_models: list[str] | None = field(default=None, init=False, repr=False)
    _remote_models_fetched_at: float | None = field(default=None, init=False, repr=False)
    _remote_models_key: tuple[str, str] | None = field(default=None, init=False, repr=False)
    _remote_models_error: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # DeepSeek client uses a compatible but distinct param type; treat as ChatClient.
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.agent = self._make_agent()
        self.sync_display_flags()
        self._bind_todo_tools()

    def sync_display_flags(self) -> None:
        from plyngent.cli.display import set_markdown_enabled, set_verbose_tool_results

        set_verbose_tool_results(self.verbose)
        set_markdown_enabled(self.markdown_enabled)

    def _workspace_key(self) -> str:
        key = normalize_workspace(self.workspace)
        if key is None:
            msg = "workspace path is required"
            raise RuntimeError(msg)
        return key

    def effective_yolo(self) -> YoloMode:
        """Resolved YOLO mode (session override or config default)."""
        if self.yolo is not None:
            return self.yolo
        return "off" if self.config.agent_config.confirm_destructive else "on"

    def soft_confirm_enabled(self) -> bool:
        """Whether destructive tools should prompt (or deny non-interactively)."""
        return self.effective_yolo() == "off"

    def set_yolo(self, mode: YoloMode) -> None:
        """Set YOLO mode; rebuild tool registry when soft-confirm hooks change."""
        prev = self.soft_confirm_enabled()
        self.yolo = mode
        if prev != self.soft_confirm_enabled():
            self.rebuild_client()

    def expire_yolo_once(self, *, quiet: bool = False) -> None:
        """If mode is ``once``, drop back to ``off`` after a user turn."""
        if self.effective_yolo() != "once":
            return
        self.set_yolo("off")
        if not quiet:
            import click

            click.secho("yolo=off (once expired)", fg="bright_black", err=True)

    def _bind_todo_tools(self) -> None:
        """Point module-level todo tools at this session stack + persist hook."""
        import asyncio

        def on_change() -> None:
            if self.session_id is None:
                return
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return
            task = loop.create_task(self.memory.update_session_todo_stack(self.session_id, self.todo_stack.to_raw()))
            # Keep a strong ref until done so the task is not GC'd mid-flight.
            self._todo_persist_tasks.add(task)
            task.add_done_callback(self._todo_persist_tasks.discard)

        set_todo_stack(self.todo_stack, on_change=on_change)

    async def persist_todo_stack(self) -> None:
        """Write the in-memory todo stack to the active session row."""
        if self.session_id is None:
            return
        _ = await self.memory.update_session_todo_stack(self.session_id, self.todo_stack.to_raw())

    async def load_todo_stack(self) -> None:
        """Load todo stack from the active session (empty if none)."""
        if self.session_id is None:
            self.todo_stack = TodoStack()
            self._bind_todo_tools()
            if hasattr(self, "agent"):
                self.agent.todo_stack = self.todo_stack
            return
        raw = await self.memory.get_session_todo_stack(self.session_id)
        self.todo_stack = TodoStack.from_raw(raw)
        self._bind_todo_tools()
        if hasattr(self, "agent"):
            self.agent.todo_stack = self.todo_stack

    def _tool_registry(self) -> ToolRegistry | None:
        if not self.tools_enabled:
            return None
        from plyngent.cli.limits import prompt_confirm_tool_async
        from plyngent.tools.danger import classify_danger

        if self.soft_confirm_enabled():
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
            todo_stack=self.todo_stack,
        )

    def rebuild_client(self) -> None:
        """Recreate client and agent after provider/model/tools change."""
        messages = list(self.agent.messages)
        persist_from = self.agent.persist_from
        # Preserve live stream toggle if agent already exists.
        if hasattr(self, "agent"):
            self.stream_enabled = self.agent.stream
        self.client = cast("ChatClient", cast("object", create_client(self.provider)))
        self.agent = self._make_agent()
        # Restore history without re-marking already-stored messages as dirty.
        self.agent.replace_messages(messages, persist_from=persist_from)
        self.sync_display_flags()
        self._bind_todo_tools()
        # Drop remote catalog when provider identity/url changed (not on model-only switch).
        if self._remote_models_key is not None and self._remote_models_key != self._models_cache_key():
            self.invalidate_remote_models()

    def _models_cache_key(self) -> tuple[str, str]:
        url = getattr(self.provider, "url", "") or ""
        return (self.provider_name, str(url))

    def invalidate_remote_models(self) -> None:
        """Drop cached remote model catalog (provider/client change)."""
        self._remote_models = None
        self._remote_models_fetched_at = None
        self._remote_models_key = None
        self._remote_models_error = None

    def seed_remote_models(self, ids: list[str]) -> None:
        """Install a freshly fetched remote catalog into the session cache."""
        self._remote_models = list(ids)
        self._remote_models_fetched_at = time.monotonic()
        self._remote_models_key = self._models_cache_key()
        self._remote_models_error = None

    def remember_session_ids(self, ids: Sequence[int]) -> None:
        """Cache session ids for Tab completion (``/resume`` / ``/delete``)."""
        self._session_id_cache = [int(i) for i in ids]

    def session_ids_for_complete(self) -> list[str]:
        """String session ids for completers (cache + current session if any)."""
        seen: set[int] = set()
        out: list[str] = []
        for sid in self._session_id_cache:
            if sid not in seen:
                seen.add(sid)
                out.append(str(sid))
        if self.session_id is not None and self.session_id not in seen:
            out.insert(0, str(self.session_id))
        return out

    def cached_remote_models(self) -> list[str] | None:
        """Return cached remote ids if still valid for the current provider."""
        if self._remote_models is None or self._remote_models_fetched_at is None:
            return None
        if self._remote_models_key != self._models_cache_key():
            return None
        age = time.monotonic() - self._remote_models_fetched_at
        if age > DEFAULT_MODELS_CACHE_TTL:
            return None
        return list(self._remote_models)

    def model_choice_ids(self, *, include_remote_cache: bool = True) -> list[str]:
        """Config plus optional cached remote ids (no network)."""
        remote = self.cached_remote_models() if include_remote_cache else None
        return model_choices_for_provider(self.provider, remote_ids=remote)

    async def ensure_remote_models(self, *, refresh: bool = False) -> list[str]:
        """Fetch ``GET /models`` (cached) and return remote ids.

        Raises RuntimeError/TypeError when the client cannot list models or
        the request fails. On failure the previous cache is left unchanged.
        """
        if not refresh:
            cached = self.cached_remote_models()
            if cached is not None:
                return cached
        if not client_supports_models(self.client):
            msg = "client does not support listing models"
            self._remote_models_error = msg
            raise TypeError(msg)
        try:
            ids = await fetch_remote_model_ids(self.client)
        except (RuntimeError, TypeError, OSError, ValueError) as exc:
            self._remote_models_error = str(exc)
            raise
        self._remote_models = list(ids)
        self._remote_models_fetched_at = time.monotonic()
        self._remote_models_key = self._models_cache_key()
        self._remote_models_error = None
        return list(ids)

    async def merged_model_choices(self, *, refresh: bool = False) -> list[str]:
        """Config plus remote catalog; remote fetch best-effort when refresh/missing."""
        remote: list[str] | None
        try:
            remote = await self.ensure_remote_models(refresh=refresh)
        except RuntimeError, TypeError, OSError, ValueError:
            remote = self.cached_remote_models()
        return model_choices_for_provider(self.provider, remote_ids=remote)

    def config_model_ids(self) -> list[str]:
        return config_model_ids(self.provider)

    def reload_config_from_disk(self) -> None:
        """Re-read TOML config and re-bind provider/model when still valid."""
        import click

        from plyngent.cli.selection import select_model, select_provider
        from plyngent.runtime import ProviderNotSupportedError
        from plyngent.tools import set_path_denylist

        self.config.reload()
        set_path_denylist(self.config.agent_config.path_denylist or None)

        selectable = self.config.selectable_providers()
        preferred_provider = self.provider_name if self.provider_name in selectable else None
        preferred_model = self.model
        try:
            pname, provider = select_provider(
                selectable,
                preferred=preferred_provider,
                interactive=False,
            )
        except (click.ClickException, ProviderNotSupportedError) as exc:
            msg = f"config reloaded but provider selection failed: {exc}"
            raise ValueError(msg) from exc
        # Empty-models providers stay recoverable until next use /models promote.
        if not provider.models and preferred_model:
            with contextlib.suppress(KeyError, ValueError):
                provider = self.config.promote_provider(pname, [preferred_model])
        try:
            model_id = select_model(provider, preferred=preferred_model, interactive=False)
        except click.ClickException:
            model_id = next(iter(sorted(provider.models.keys()))) if provider.models else preferred_model

        self.provider_name = pname
        self.provider = provider
        self.model = model_id
        self.rebuild_client()
        self.invalidate_remote_models()

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

    def persist_models_to_config(
        self,
        *,
        mode: Literal["current", "catalog"],
        catalog_ids: Sequence[str] | None = None,
    ) -> Path:
        """Merge model id(s) into TOML for the current provider and write disk.

        *mode* ``current``: ensure :attr:`model` is in the provider catalog.
        *mode* ``catalog``: union *catalog_ids* (or empty) into the catalog.

        Returns the config path written. Raises ``OSError`` / ``ValueError`` /
        ``KeyError`` on failure.
        """
        if mode == "current":
            self.provider = self.config.ensure_model(self.provider_name, self.model)
        else:
            ids = list(catalog_ids) if catalog_ids is not None else []
            self.provider = self.config.merge_models(self.provider_name, ids)
        self.config.write()
        return self.config.path

    def _try_set_provider(self, pname: str) -> bool:
        import click

        from plyngent.cli.selection import select_provider
        from plyngent.runtime import ProviderNotSupportedError

        if pname not in self.config.selectable_providers():
            return False
        try:
            name, provider = select_provider(
                self.config.selectable_providers(),
                preferred=pname,
                interactive=False,
            )
        except click.ClickException, ProviderNotSupportedError:
            return False
        # Session resume: seed empty recoverable with remembered model if any.
        if not provider.models and self.model:
            try:
                provider = self.config.promote_provider(name, [self.model])
            except KeyError, ValueError:
                return False
        if name != self.provider_name or provider is not self.provider:
            self.provider_name = name
            self.provider = provider
            return True
        return False

    def _try_set_model(self, model_id: str) -> bool:
        token = model_id.strip()
        if not token:
            return False
        if token != self.model:
            self.model = token
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
        if session.sid not in self._session_id_cache:
            self._session_id_cache.insert(0, session.sid)
        self.todo_stack = TodoStack()
        self.agent = self._make_agent()
        self._bind_todo_tools()

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
        if session_id not in self._session_id_cache:
            self._session_id_cache.insert(0, session_id)
        if self.apply_session_llm(row):
            self.rebuild_client()
        else:
            self.agent = self._make_agent()
        await self.agent.load_history()
        await self.load_todo_stack()

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
        await self.load_todo_stack()
        _ = await self.memory.touch_session(latest.sid)
        return "resume"

    async def compact_to_new_session(self, *, name: str | None = None) -> tuple[int, int, str]:
        """Soft-compact + model-summarize current history into a new workspace session.

        Returns ``(old_session_id, new_session_id, summary)``.
        Todo stack is carried into the new session.
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
        carried_todos = self.todo_stack.to_raw()
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
        for message in seed:
            _ = await self.memory.append_message(new_id, message)
        # Reload from DB so RAM matches stored rows and the persist cursor is correct
        # (assigning messages alone left _persist_from at 0 and broke later checkpoints).
        await self.agent.load_history()
        if not self.agent.messages:
            msg = f"compact session {new_id} has no messages after seed"
            raise RuntimeError(msg)
        # Carry open/closed todos so multi-step work survives compact.
        self.todo_stack = TodoStack.from_raw(carried_todos)
        self._bind_todo_tools()
        self.agent.todo_stack = self.todo_stack
        await self.persist_todo_stack()
        return old_id, new_id, summary
