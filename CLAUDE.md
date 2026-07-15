# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plyngent is an LLM chat and agent toolkit (Python 3.14+, PDM-managed). Early development: protocol clients, config, async memory, and a runtime client factory exist. Agent, CLI, web, tools, and router are still planned/stubbed.

## Commands

```bash
pdm install          # first-time dependency setup
pdm sync             # sync after pulling changes

pdm run basedpyright .   # type checking (basedpyright, "recommended" strictness)
pdm run ruff check .     # linting
pdm run ruff format .    # formatting
pdm run pytest           # tests (pytest-asyncio auto mode)
```

## Architecture

### Data modeling: `msgspec.Struct`

All protocol models use `msgspec.Struct` — not dataclasses, not Pydantic. Optional fields use `msgspec.field(default=UNSET)` / `default=UNSET` with type `T | Unset`.

`typedef.py`: `Unset = UnsetType` (plain assignment so msgspec recognizes it; do **not** use PEP 695 `type Unset = ...`). Multi-struct unions must be **tagged** via `tag_field` / `tag` (e.g. `role`, `type`) for decode. `JSONSchema` is `dict[str, Any]`.

### Protocol layering (`lmproto/`)

```
lmproto/openai_compatible/     ← Base: model, config, client
lmproto/deepseek/openai_compat/ ← Extends base via inheritance + extra fields
```

- **`openai_compatible/model.py`** — tagged chat messages (`SystemChatMessage`, `UserChatMessage`, …), tools, request/response, streaming chunks.
- **`openai_compatible/client.py`** — `BaseOpenAIClient` / `OpenAIClient` via `niquests` async + SSE.
- **`openai_compatible/config.py`** — `OpenAIConfig` (token + base URL).
- **DeepSeek** — `DeepseekOpenAIClient`; models add `reasoning_content`, `prefix`, `ThinkingOptions`.

### Config (`config/`)

TOML load/store (`ConfigStore`): `[providers]` tagged union presets, `[database]` section. Default path via platformdirs.

### Runtime (`runtime/`)

`create_client(provider)` maps config `Provider` → protocol client. OpenAI / openai-compatible / deepseek(openai convention) supported; anthropic and deepseek anthropic convention raise `ProviderNotSupportedError`.

### Memory (`memory/`)

Async SQLAlchemy + aiosqlite. `MemoryStore`: schema init (+ lightweight SQLite `ALTER` for new columns), default local user, sessions (bound to `workspace` path), messages stored as msgspec chat message JSON.

### Agent (`agent/`)

- **`ChatClient`** Protocol for `chat_completions`.
- **`@tool` / `ToolRegistry`**: decorator infers JSON Schema from type hints; execute tools by name.
- **`run_chat_loop`**: multi-round tool loop; default **streaming** text deltas + stream tool-call merge; parallel tools; tool-result char budget; soft context compact on request; cooperative cancel points; optional `on_limit`.
- **`ChatAgent`**: optional `MemoryStore` (persist on success only); `stream`; system prompt; `pending_retry_text` + `retry()`.
- **`/compact`**: soft-compact tool dumps → model summary (no tools) → **new** session seeded with summary message.
- Events: text_delta, assistant_message, tool_call/result, max_rounds, **error** (`retryable`/`source`), **cancelled** (`reason`).
- Config ``[agent]``: `system_prompt`, `max_tool_result_chars`, `parallel_tools`, `confirm_destructive`, `path_denylist`, `max_context_chars`.

### Tools (`tools/`)

Module-level `@tool` handlers. Call `set_workspace_root()` before use.

- **`workspace`**: path resolve under root; path substring denylist; command basename denylist; clearer deny messages.
- **`file`**: `read_file`, `write_file`, `listdir`, `tree`, `edit_replace`, `edit_lineno` (1-based range), `copy_path` / `move_path` / `delete_path`.
- **`process`**: `run_command` (argv, no shell, timeout, optional stdin/env); PTY `open_pty` / `read_pty` / `write_pty` / `close_pty` (**Unix only**: `pty`+`fork`).
- PTY: structured status (`alive`/`exit_code`/`data`); `read_pty(..., until=)`; session limit/idle TTL/output budget; close SIGTERM→SIGKILL.
- CLI limit hooks: interactive confirm to raise tool-loop rounds, PTY session cap, or PTY output budget.
- Destructive confirms: `classify_danger` + `ToolRegistry(on_confirm=…)`; CLI default deny; config `confirm_destructive` / `path_denylist`.
- **`DEFAULT_TOOLS`**: file + process tool list for a `ToolRegistry`.

### CLI (`cli/`)

Click app + readline REPL. Entry: `plyngent` / `python -m plyngent`.

- **`plyngent chat`**: provider/model selection (flags or interactive), SQLite sessions via config `[database]` (file DB under user data if unset/`:memory:`), sessions bound to workspace dir; resumes **most recently updated** session for cwd/`--workspace` by default (`--new` / `--session`).
- Slash: `/history`, `/sessions` (newest first), `/resume [id]`, `/compact`, `/status` (incl. context char estimate), `/rounds`, `/retry`, …
- Explicit `/resume` or `--session` from another workspace prompts: **keep** session path, **update** binding to current, or **abort**.
- Failed/cancelled turns: not written to DB; Ctrl+C cancels the in-flight turn task; **TTY confirms** (max-rounds / destructive tools) run off-loop via `asyncio.to_thread` + pause cancel so prompts do not abort the turn; auto-retry 10s/20s/30s; `/retry` manual.
- **`plyngent providers`**: list config providers.
- **`plyngent config path|edit`**: print or open config in `$EDITOR` (`shlex`-split, e.g. `codium --wait`).
- If no providers and `$EDITOR` is set, chat/providers prompt to edit config then reload.
- Tools default on (`--tools` / `--no-tools`); workspace defaults to cwd; `--max-rounds` default 32.
- Readline/editline: Tab completion; input history file under platformdirs user data (`repl_history`).

### Composition utility: `Forward` descriptor

`utils/components.py` — `Forward[T]` / `forward()` for attribute forwarding on composed objects.

### Type annotations are mandatory

Basedpyright `recommended`. Ruff includes `ANN` (private return types `ANN202` ignored). Prefer PEP 695 aliases except where msgspec requires plain assignment (`Unset`).

## Roadmap notes (single-user → platform)

- **Phase D (context quality)**: soft char budget, request compact, `/compact`, richer errors/cancel, workspace sessions — **current**. Context size is **char estimate only** until usage lands.
- **No local tokenizer stage** for now. Prefer **API usage v2** later.
- **Phase E**: tooling depth (grep/glob, edit_patch, optional git).
- **Phase F (providers + usage v2)**: capture response/stream `usage` (prompt/completion/total); session/turn totals; `/status` or end-of-turn line; optional cost. Optional later: tokenizer-backed estimates if needed.
- **Phase G–H**: CLI polish, hardening; then multi-tenant platform (`router/`, auth, sandboxed tools).

## Commit messages

Scoped commit messages, not conventional commits:

```
<scope>: <brief description>
```

Examples: `deps: add fastapi`, `core/mq: fix incorrect message sending`, `router: add routing service for xxx`, `test/webserver: change test client`, `ci/lint: run ruff style check`.

Check `git log` for the full convention.
