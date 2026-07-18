# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plyngent is an LLM chat and agent toolkit (Python 3.14+, PDM-managed). Single-user CLI is usable: protocol clients, config, async memory, agent tool loop, workspace tools, and REPL/one-shot chat. Multi-tenant `router/` / web remain Phase H.

## Commands

```bash
pdm install          # first-time dependency setup
pdm sync             # sync after pulling changes

pdm run ruff check .           # linting
pdm run ruff format .          # apply formatting
pdm run ruff format --check .  # CI: fail if unformatted (do not skip)
pdm run basedpyright .         # type checking (basedpyright, "recommended" strictness)
pdm run pytest                 # tests (pytest-asyncio auto mode)
```

GitHub Actions runs `ruff check`, `ruff format --check`, `basedpyright`, then `pytest`. Format locally with `pdm run ruff format .` so CI does not fail after publish.

## Architecture

### Data modeling: `msgspec.Struct`

All protocol models use `msgspec.Struct` — not dataclasses, not Pydantic. Optional fields use `msgspec.field(default=UNSET)` / `default=UNSET` with type `T | Unset`.

`typedef.py`: `Unset = UnsetType` (plain assignment so msgspec recognizes it; do **not** use PEP 695 `type Unset = ...`). Multi-struct unions must be **tagged** via `tag_field` / `tag` (e.g. `role`, `type`) for decode. `JSONSchema` is `dict[str, Any]`.

### Protocol layering (`lmproto/`)

```
lmproto/openai_compatible/     ← Base: model, config, client
lmproto/openai/                ← OpenAI platform: Responses + chat (extends base)
lmproto/deepseek/openai_compat/ ← Extends base via inheritance + extra fields
```

- **`openai_compatible/model.py`** — tagged chat messages (`SystemChatMessage`, `UserChatMessage`, …), tools, request/response, streaming chunks.
- **`openai_compatible/client.py`** — `BaseOpenAIClient` / `OpenAICompatibleClient` via `niquests` async + SSE; `chat_completions`, `models` only (`OpenAIClient` is a compat alias).
- **`openai_compatible/config.py`** — `OpenAIConfig` (token + base URL).
- **`openai/model.py`** — OpenAI Responses API (`ResponsesCreateParam`, `Response`, function_call items, stream events).
- **`openai/client.py`** — platform `OpenAIClient`: chat completions + `responses` / `get_response` / `delete_response`.
- **DeepSeek** — `DeepseekOpenAIClient`; models add `reasoning_content`, `prefix`, `ThinkingOptions`. Config default model ids: `deepseek-v4-flash`, `deepseek-v4-pro`.

### Config (`config/`)

TOML load/store (`ConfigStore`): `[providers]` tagged union presets, `[database]` section. Default path via platformdirs.

### Runtime (`runtime/`)

`create_client(provider)` maps config `Provider` → protocol client. OpenAI → `lmproto.openai.OpenAIClient` (Responses-capable); openai-compatible / deepseek(openai convention) → chat-completions clients; anthropic and deepseek anthropic convention raise `ProviderNotSupportedError`.

### Memory (`memory/`)

Async SQLAlchemy + aiosqlite. `MemoryStore`: schema init (+ lightweight SQLite `ALTER` for new columns), default local user, sessions (bound to `workspace` path; optional `provider_name`/`model`), messages stored as msgspec chat message JSON.

### Agent (`agent/`)

- **`ChatClient`** Protocol for `chat_completions` (agent history stays chat-shaped).
- **OpenAI Responses integration**: `ResponsesChatClient` adapts platform `OpenAIClient.responses` to `ChatClient`; selected automatically for `OpenAIProvider` in `create_client`. Compat/DeepSeek stay on chat completions.
- **Provider-side tools**: `OpenAIProvider.provider_tools` (list of dicts; default `[{type="web_search"}]` when omitted; `[]` disables) merged into Responses `tools` alongside local function tools; never executed by `ToolRegistry`.
- **`@tool` / `ToolRegistry`**: decorator infers JSON Schema from type hints; execute tools by name.
- **`run_chat_loop`**: multi-round tool loop; default **streaming** text deltas + stream tool-call merge; parallel tools; tool-result char budget; soft context compact on request (**API-calibrated** after first usage when available); cooperative cancel points; optional `on_limit`.
- **`ChatAgent`**: optional `MemoryStore` (user message persisted immediately; **completed tool batches checkpointed** mid-turn; unfinished assistant suffix rolled back on failure); `stream`; system prompt; `retry()` continues incomplete turns (user-only **or** after committed tools — does not re-run those tools).
- **`/compact`**: soft-compact tool dumps → model summary (no tools) → **new** session seeded with summary message.
- Events: text_delta, **reasoning_delta**, assistant_message, tool_call/result, max_rounds, **error** (`retryable`/`source`), **cancelled** (`reason`), **usage** (`TokenUsage`).
- Usage: API `usage` from completions (stream with `include_usage`); **char≈token fallback** (~4 chars/token) when omitted; **context size** = last request ``prompt_tokens`` (API preferred); `last_turn_usage` / `session_usage` are **billed sums** (tool rounds re-send history); CLI end-of-turn + `/status`.
- Config ``[agent]``: `system_prompt`, `max_tool_result_chars`, `parallel_tools`, `confirm_destructive`, `path_denylist`, `max_context_tokens` (default 200k est. tokens).

### Tools (`tools/`)

Module-level `@tool` handlers. Call `set_workspace_root()` before use.

- **`workspace`**: path resolve under root; path substring denylist; command basename denylist; clearer deny messages.
- **`file`**: `read_file`, `write_file`, `listdir`, `tree` (VCS + default noise dirs + optional `skip_dirs` / denylist walk), `glob_paths`, `grep_files` (regex, skip VCS/binary), `edit_replace`, `edit_lineno` (1-based range), `copy_path` / `move_path` / `delete_path`.
- **`process`**: `run_command` (argv, no shell, timeout, optional stdin/env); PTY `open_pty` / `read_pty` / `write_pty` / `close_pty` (POSIX: `pty`+`fork`; Windows: ConPTY via `pywinpty` env marker dep).
- PTY: backend in `pty_backend.py`; structured status (`alive`/`exit_code`/`data`); `read_pty(..., until=)`; session limit/idle TTL/output budget; close terminate→kill.
- CLI limit hooks: interactive confirm to raise tool-loop rounds, PTY session cap, or PTY output budget.
- Destructive confirms: `classify_danger` + `ToolRegistry(on_confirm=…)`; CLI default deny; config `confirm_destructive` / `path_denylist`. Session YOLO: `/yolo on|off|once` and `--yes` (skip soft confirms; hard denylists unchanged; `once` expires after the next user turn).
- **`vcs`**: read-only VCS tools (`vcs_kind` / `vcs_status` / `vcs_diff` / `vcs_log` / `vcs_branch`) via `VcsBackend` protocol; **git** implemented; detectors are pluggable for other systems.
- **`chat`**: human prompts as tools — `ask_user_line` / `ask_user_choice` / `ask_user_form` (shared `prompting` core).
- **`todo`**: nested session sub-task stack (frames) — `todo_list` / `todo_push` (multi-title = new frame) / `todo_pop` (leave frame) / `todo_update` / `todo_clear`; pattern push[T1,T2]→push[T1.1…]→pop→push[T2.1…]; stored on session row; agent injects a developer review message if open todos and no todo tool use in the turn.
- **`DEFAULT_TOOLS`**: file + process + vcs + chat + todo tool list for a `ToolRegistry`.

### Prompting (`prompting.py`)

Shared interactive I/O: `ask` / `choose` / `form` / `confirm` with pluggable backend; non-TTY uses defaults or errors. CLI limit/confirm hooks and chat tools both use this. Async helpers serialize prompts (`run_prompt_async`).

### CLI (`cli/`)

Click app + readline REPL. Entry: `plyngent` / `python -m plyngent`.

- **`plyngent chat`**: provider/model (flags or interactive; Tab via readline in `prompting`); sessions store `provider_name`/`model` and restore on resume; SQLite via `[database]` (file DB under user data if unset/`:memory:`); workspace-bound; resume latest for cwd/`--workspace` by default (`--new` / `--session`). One-shot: `-p/--prompt` and non-TTY stdin; exit codes 0/1/2/3; `--yes` (YOLO on), `--stream/--no-stream`, `--quiet`. Root `--log-level`.
- Slash: Click group in `cli/slash.py` + `awaitlet` for async work; Tab completer from registry + ParamType `shell_complete`. Multiline `"""` … `"""`; `/edit` via `$EDITOR`. `/yolo on|off|once` for soft destructive confirms. `/model --persist` / `/models --persist` write model catalog entries into TOML. `/todos` for human show/push/pop/clear of the todo stack.
- Explicit `/resume` or `--session` from another workspace prompts: **keep** / **update** / **abort**.
- Failed/cancelled turns: user kept; **committed tool rounds kept** (side effects not re-run on `/retry`); only unfinished assistant rolled back; Ctrl+C cancels; TTY confirms off-loop; auto-retry 10s/20s/30s then `/retry`.
- **`plyngent providers`**, **`config path|edit`**. No providers + `$EDITOR` → optional edit then reload.
- Tools default on; workspace defaults to cwd; `--max-rounds` default 32. Readline history under platformdirs (`repl_history`). PTY: `close_all` on chat exit.

### Composition utility: `Forward` descriptor

`utils/components.py` — `Forward[T]` / `forward()` for attribute forwarding on composed objects.

### Type annotations are mandatory

Basedpyright `recommended`. Ruff includes `ANN` (private return types `ANN202` ignored). Prefer PEP 695 aliases except where msgspec requires plain assignment (`Unset`).

## Roadmap notes (single-user → platform)

- **Phase D (context quality)**: soft char budget, request compact, `/compact`, richer errors/cancel, workspace sessions. Context size is **char estimate** plus optional API usage when reported.
- **No local tokenizer stage** for now.
- **Phase E**: tooling depth (grep/glob, VCS backends; prefer `edit_replace` / `edit_lineno` over model-generated patches).
- **Phase F (providers + usage v2)**: API `usage` + char-based estimate fallback; session/turn totals; `/status` + end-of-turn. Optional later: cost, real tokenizer.
- **Phase G (CLI polish + hardening)** — single-user only; multi-tenant stays Phase H.

  **Done**
  - G0: `prompting` (`ask`/`choose`/`form`/`confirm`) + chat tools (`ask_user_line`/`ask_user_choice`/`ask_user_form`)
  - G1: `ReasoningDeltaEvent`, `/stream`, `/verbose`
  - G2: `/rename`, `/delete` (confirm), `/export md|json`
  - G2.5: Click slash registry (`cli/slash.py`) + `awaitlet` for sync Click / async memory; auto `/help`; completer from `slash.list_commands`
  - G3: multiline `"""` … `"""` input (`cli/input_text.py`); `/edit` via `$EDITOR` (`edit_text_in_editor`)
  - G4: `plyngent chat -p/--prompt` (+ non-TTY stdin); exit codes 0/1/2/3; `--yes` / non-interactive confirm deny; `--stream/--no-stream`, `--quiet`
  - G5: PTY master FD non-inheritable; `read_pty`/`close_pty` via `to_thread`; `PtyManager.close_all()` on chat exit; `--log-level`; clearer invalid TOML errors; export/status stay secret-free
  - G6: README, `doc/plyngent.example.toml`, CLAUDE overview/CLI notes

  Phase G complete for single-user CLI polish. Next roadmap work is Phase H or optional F (cost/tokenizer).

  **PTY / process model (decision)**

  Today `PtyManager` (`tools/process/pty_session.py`) is **in-process**: `pty.openpty` + `os.fork` → child `execvp`, parent holds master FD; `read_pty`/`close_pty` run via `to_thread`. Session registry is process-global. Safe enough for single-user CLI if the child path stays fork-then-exec only.

  | Question | Answer |
  |----------|--------|
  | Separate PTY supervisor process so the main app is safer with threads/greenlets? | **Not for Phase G.** Defer to **Phase H** (sandbox / multi-tenant) or if we hit real FD-leak / freeze bugs. |
  | Why not now? | Fork-then-exec is already the right shape; rewrite cost (IPC, lifecycle, tests) dwarfs single-user CLI risk. awaitlet greenlets are slash-only; PTY is not forked from a worker thread today. |
  | G5 (done) | Master FD non-inheritable; `read_pty`/`close_pty` via `to_thread`; chat exit `close_all`; fork stays on loop/main thread. |
  | Phase H | Optional PTY helper process (JSON/Unix socket), or subprocess+PTY with asyncio reaping; sandboxed tools, multi-session isolation. |

- **Phase H**: multi-tenant platform (`router/`, auth, sandboxed tools, web). Optional **out-of-process PTY host** if isolation is required.

## Commit messages

Scoped commit messages, not conventional commits:

```
<scope>: <brief description>
```

Examples: `deps: add fastapi`, `core/mq: fix incorrect message sending`, `router: add routing service for xxx`, `test/webserver: change test client`, `ci/lint: run ruff style check`.

Check `git log` for the full convention.
