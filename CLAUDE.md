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

Async SQLAlchemy + aiosqlite. `MemoryStore`: schema init, default local user, sessions, messages stored as msgspec chat message JSON.

### Agent (`agent/`)

- **`ChatClient`** Protocol for `chat_completions`.
- **`@tool` / `ToolRegistry`**: decorator infers JSON Schema from type hints; execute tools by name.
- **`run_chat_loop`**: multi-round tool loop, yields `AgentEvent` stream.
- **`ChatAgent`**: wrapper with optional `MemoryStore` bind (load/append messages).

### Tools (`tools/`)

Module-level `@tool` handlers. Call `set_workspace_root()` before use.

- **`workspace`**: path resolve under root; path substring denylist; command basename denylist.
- **`file`**: `read_file`, `write_file`, `listdir`, `edit_replace` (first occurrence).
- **`process`**: `run_command` (argv, no shell, timeout); minimal PTY `open_pty` / `read_pty` / `close_pty`.
- **`DEFAULT_TOOLS`**: file + process tool list for a `ToolRegistry`.

### CLI (`cli/`)

Click app + readline REPL. Entry: `plyngent` / `python -m plyngent`.

- **`plyngent chat`**: provider/model selection (flags or interactive), SQLite sessions via config `[database]`, `/help` slash commands.
- **`plyngent providers`**: list config providers.
- **`plyngent config path|edit`**: print or open config in `$EDITOR` (`shlex`-split, e.g. `codium --wait`).
- If no providers and `$EDITOR` is set, chat/providers prompt to edit config then reload.
- Tools default on (`--tools` / `--no-tools`); workspace defaults to cwd.
- Readline: Tab completion for slash commands/args; history file under platformdirs user data (`repl_history`).

### Composition utility: `Forward` descriptor

`utils/components.py` — `Forward[T]` / `forward()` for attribute forwarding on composed objects.

### Type annotations are mandatory

Basedpyright `recommended`. Ruff includes `ANN` (private return types `ANN202` ignored). Prefer PEP 695 aliases except where msgspec requires plain assignment (`Unset`).

## Commit messages

Scoped commit messages, not conventional commits:

```
<scope>: <brief description>
```

Examples: `deps: add fastapi`, `core/mq: fix incorrect message sending`, `router: add routing service for xxx`, `test/webserver: change test client`, `ci/lint: run ruff style check`.

Check `git log` for the full convention.
