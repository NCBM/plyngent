# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plyngent is an LLM chat and agent toolkit (Python 3.14+, PDM-managed). Single-user CLI is usable: protocol clients, config, async memory, agent tool loop, workspace tools, skill discovery, and REPL/one-shot chat. Multi-tenant `router/` / web remain Phase H.

## Commands

```bash
pdm install          # first-time dependency setup
pdm sync             # sync after pulling changes

pdm run ruff check .           # linting
pdm run ruff format .          # apply formatting
pdm run ruff format --check .  # CI: fail if unformatted (do not skip)
pdm run basedpyright .         # type checking (basedpyright, "recommended" strictness)
pdm run pytest                 # tests (pytest-asyncio auto mode)

# Commit gateway (prek — https://prek.j178.dev/): ruff check/format + basedpyright
uv tool install prek           # once
prek install                   # wire git pre-commit hook (once per clone)
prek run --all-files           # run all hooks on demand
```

GitHub Actions runs `ruff check`, `ruff format --check`, `basedpyright`, then `pytest`. Local commits run the same checks via `prek.toml` so `ruff format` is not forgotten.

## Architecture

### Data modeling: `msgspec.Struct`

All protocol models use `msgspec.Struct` — not dataclasses, not Pydantic. Optional fields use `msgspec.field(default=UNSET)` / `default=UNSET` with type `T | Unset`.

`typedef.py`: `Unset = UnsetType` (plain assignment so msgspec recognizes it; do **not** use PEP 695 `type Unset = ...`). Multi-struct unions must be **tagged** via `tag_field` / `tag` (e.g. `role`, `type`) for decode. `JSONSchema` is `dict[str, Any]`.

### Protocol layering (`lmproto/`)

```
lmproto/openai_compatible/     ← Base: model, config, client
lmproto/openai/                ← OpenAI platform: Responses + chat (extends base)
lmproto/anthropic/             ← Anthropic Messages: model, config, client
lmproto/deepseek/openai_compat/ ← Extends base via inheritance + extra fields
lmproto/deepseek/responses/     ← DeepSeek Responses client (extends OpenAI Responses client; SSE stops at terminal events, no `[DONE]`)
lmproto/deepseek/anthropic/     ← DeepSeek Anthropic client (extends AnthropicClient; no `GET /models`)
```

- **`openai_compatible/model.py`** — tagged chat messages (`SystemChatMessage`, `UserChatMessage`, …), tools, request/response, streaming chunks.
- **`openai_compatible/client.py`** — `BaseOpenAIClient` / `OpenAICompatibleClient` via `niquests` async + SSE; `chat_completions`, `models` only.
- **`openai_compatible/config.py`** — `OpenAIConfig` (token + base URL).
- **`openai/model.py`** — OpenAI Responses API (`ResponsesCreateParam`, `Response`, function_call items, stream events).
- **`openai/client.py`** — platform `OpenAIClient` (`kind="responses"`): chat completions + `responses` / `get_response` / `delete_response`.
- **`anthropic/model.py`** — Anthropic Messages request/response + SSE event models.
- **`anthropic/client.py`** — `AnthropicClient` (`kind="messages"`): `POST /messages` + `GET /models`.
- **DeepSeek** — `DeepseekOpenAIClient` (`kind="chat_completions"`); models add `reasoning_content`, `prefix`, `ThinkingOptions`. Config default model ids: `deepseek-v4-flash`, `deepseek-v4-pro`. `DeepseekResponsesClient` (`kind="responses"`) extends the OpenAI Responses client for `convention = "responses"` (SSE stops at terminal events — DeepSeek sends no `[DONE]`). `DeepseekAnthropicClient` (`kind="messages"`) extends `AnthropicClient` for `convention = "anthropic"` (base `https://api.deepseek.com/anthropic`; `models()` returns `[]`).

### Config (`config/`)

TOML load/store (`ConfigStore`): `[providers]` tagged union presets, `[database]` section. Default path via platformdirs. Providers may set `timeout` as a float or `{ connect, read }` (`HttpTimeoutConfig`); omitted → product defaults (10s connect / 600s read). `DeepseekProvider.convention` (Literal: `""`/`"openai"`/`"anthropic"`/`"responses"`) selects the DeepSeek API surface — `"responses"` → OpenAI Responses, `"anthropic"` → Anthropic Messages on `https://api.deepseek.com/anthropic` (default url follows the convention); a per-model `ModelConfig.convention` overrides it (empty = inherit). `[mcp]` (`McpConfig`): `servers.<name>` stdio server definitions (`McpServerConfig`: `command`/`args`/`env`/`cwd`/`timeout`/`read_only`, `url` reserved for streamable HTTP) + `disable` gate; defining a server opts it in. `[skills]` (`SkillsConfig`): `paths` (explicit roots first), `discover` (`plyngent` user dir + `claude` dirs) with `include_project_roots`, `inject_catalog` / `max_catalog_skills`, `enabled`, and `allow_write` (standing write grant); a malformed section degrades to defaults like `[agent]` / `[mcp]`.

### Runtime (`runtime/`)

`create_client(provider)` maps config `Provider` → protocol client by effective preset (model-level `preset`/`url` overrides apply). `normalize_http_timeout` feeds OpenAI-compatible configs and Anthropic. Mapping: `openai` → `OpenAIClient` (agent uses Responses), `openai-compatible` / `deepseek` → chat-completions clients (deepseek with effective `convention = "responses"` → `DeepseekResponsesClient`, `"anthropic"` → `DeepseekAnthropicClient`), `anthropic` → `AnthropicClient`.

MCP (`runtime/mcp_client.py`): `McpManager` owns one `McpServerConnection` per `[mcp]` server — a spawned stdio subprocess speaking newline-delimited JSON-RPC 2.0: initialize handshake, `tools/list` (cursor pagination), `tools/call`, `notifications/tools/list_changed` → `tools_stale`. Spawn/handshake failures record `connection.error` (status string), never raise; `ensure_started` / `restart(config)` / `aclose` / `refresh_tools`; `statuses()` → `(server, status, tool_count)` rows; `instructions()` → `(server, text)` for every connected server that returned optional `InitializeResult.instructions` (usage guidance captured per connection); per-request timeout from config (default 30s).

### Skills (`skills/`)

Discovery and parsing, independent of the agent: `build_roots(config, workspace=…, user_config_dir=…, home=…)` resolves `[skills]` into ordered `SkillRoot`s (`config` explicit paths → `plyngent` user dir → `claude` user/project dirs), `SkillStore` scans each root for directories with a `SKILL.md` (a root may itself be one), parses the frontmatter subset (`frontmatter.py`), and caches the records against the directory list and its `SKILL.md` timestamps. `skills` returns the visible records sorted by name (first root wins), `shadowed()` the losing copies, `find(name)` exact-then-case-insensitive, `default_write_root()` our own directory (or the first root), `reload()` forces a rescan; `resolve_inside` / `resolve_skill_file` keep every read and write inside a skill directory (absolute paths and symlink escapes are refused). Records carry `issues` (unusable declared name, no description, unparseable frontmatter) so a repaired third-party skill stays visible instead of being silently fixed.

### Memory (`memory/`)

Async SQLAlchemy + aiosqlite. `MemoryStore`: schema init (+ lightweight SQLite `ALTER` for new columns; `PRAGMA user_version` 4), default local user, sessions (bound to `workspace` path; optional `provider_name`/`model`; `todo_stack` + `access_grants` JSON columns), messages stored as msgspec chat message JSON.

### Agent (`agent/`)

- **Kind-based dispatch**: clients expose `kind` (`chat_completions` | `responses` | `messages`). Agent history stays chat-completions-shaped; transport adapters return synthetic `ChatCompletionResponse` / stream chunks.
- **`responses_bridge` / `responses_dispatch`**: chat ↔ OpenAI Responses; stream text/reasoning deltas; tool calls + usage on `response.completed`.
- **`messages_bridge` / `messages_dispatch`**: chat ↔ Anthropic Messages; system fold; tool_use/tool_result; stream text + tool argument fragments.
- **Provider-side tools**: `OpenAIProvider.provider_tools` (list of dicts; default `[{type="web_search"}]` when omitted; `[]` disables) threaded via `ChatAgent`/`run_chat_loop` into Responses only; never executed by `ToolRegistry`.
- **`@tool` / `ToolRegistry`**: decorator infers JSON Schema from type hints; execute tools by name. Before running, `execute` answers a call that omits a required argument with the accepted names (`_missing_required_error`) — the handler's bare `TypeError` never reaches the model, and no confirm prompt fires for a call that cannot run. Shape mistakes inside a *present* argument are the tool's own job (see the `chat` bullet).
- **`run_chat_loop`**: multi-round tool loop; default **streaming** text deltas + stream tool-call merge; parallel tools; tool-result char budget; soft context compact on request (**API-calibrated** after first usage when available); cooperative cancel points; optional `on_limit`.
- **`ChatAgent`**: optional `MemoryStore` (user message persisted immediately; **completed tool batches checkpointed** mid-turn; unfinished assistant suffix rolled back on failure); `stream`; system prompt; `provider_tools`; `retry()` continues incomplete turns (user-only **or** after committed tools — does not re-run those tools).
- **`/compact`**: soft-compact tool dumps → model summary (no tools) via the same kind dispatch → **new** session seeded with summary message. Soft-compact keeps the last **12** tool results full-size by default (`DEFAULT_RECENT_TOOL_RESULTS`); older tool payloads shrink first.
- Events: text_delta, **reasoning_delta**, assistant_message, tool_call/result, max_rounds, **error** (`retryable`/`source`), **cancelled** (`reason`), **usage** (`TokenUsage`).
- Usage: API `usage` from completions (stream with `include_usage`); **char≈token fallback** (~4 chars/token) when omitted; **context size** = last request ``prompt_tokens`` (API preferred); `last_turn_usage` / `session_usage` are **billed sums** (tool rounds re-send history); CLI end-of-turn + `/status`.
- Config ``[agent]``: `system_prompt` (persona; default `DEFAULT_SYSTEM_PROMPT`; `""` omits persona), `tool_directives` (tool playbook; default `DEFAULT_TOOL_DIRECTIVES`; `""` omits playbook; both empty → no system), `mcp_instructions` (fold connected MCP servers' initialize ``instructions`` into the system prompt when tools on; default true), `[skills] inject_catalog` (fold the skill catalog — name/source/description only — the same way), `allow_paths` (static out-of-workspace pre-allow `path → read|write|exec`; never prompts), `auto_continue_limits` (auto-raise tool/PTY limits without prompting; default false), `max_tool_result_chars`, `parallel_tools`, `confirm_destructive`, `path_denylist`, `max_context_tokens` (default 200k est. tokens). Truncated tool results carry a model hint (`[Truncated (N chars max; M omitted). You may ask for user to increase the limit via configuration file.]`) so the model can ask the user to raise `max_tool_result_chars`; request-time compact shrinks (`hint=False`) keep the terse `...[truncated N characters]` marker.

### Tools (`tools/`)

Module-level `@tool` handlers. Call `set_workspace_root()` before use.

- **`workspace`**: path resolve under primary root **or** temporary allowlist roots **or** a directory-access grant (mode-aware: `read` < `write` < `exec`, default required `write`; a file grant matches only that file); path substring denylist (wins over grants); command basename denylist (CLI: timed human allow override, default deny on timeout; **not** skipped by YOLO); escape error hints `request_directory_access`.
- **`access`** (`tools/access.py`): `request_directory_access` (LOCAL|INSTANCE|SESSION) asks the host hook for out-of-workspace access; grants are one `path → mode` pair in one of three stores — `WorkspacePolicy.config_allow` (static `[agent].allow_paths`), `SessionState.access_grants` (human-approved, persisted on the session row), `WorkspacePolicy.yolo_allow` (`--yes`/`/yolo`, process only). `AccessDecision(mode, persist)` from the hook; denylisted paths can never be granted; cap `MAX_ACCESS_GRANTS`. Not a sandbox: `exec` grants (and any command) reach the whole FS.
- **`file`**: `read_file` (`with_lineno`, `max_chars` default 32k; results start with a 1-based `L{begin}-{end}` range), `write_file`, `listdir`, `tree` (model picks `format=markdown` default / `flat` paths / `decorated`; VCS + default noise dirs + optional `skip_dirs` / denylist walk), `glob_paths`, `grep_files` (Python `re` per line, not a glob; literal file/dir `path`, no glob expansion; skips VCS/binary), `edit_replace` (`max_replaces`, reports remaining matches), `edit_lineno` (1-based range; only lines read via `with_lineno`; read state resets on any file write/edit/copy/move/delete), `copy_path` / `move_path` / `delete_path`, `new_temporary_workspace` (system temp allowlist; cleanup on chat exit; no session rebind), `get_truncated` (resume any truncated result via its `truncate_token=...`).
- **`process`**: `run_argv` (argv, no shell, timeout, optional stdin/env); `run_argv_batch` (serial steps, `pipe_out` on provider, `mix_stderr`, `stop_on_error`); PTY `open_pty` / `read_pty` / `write_pty` (literal) / `write_pty_keys` (escapes) / `ask_into_pty` (human→PTY only; `secret` no-echo; answer never in tool result) / `close_pty` (POSIX: `pty`+`fork`; Windows: ConPTY via `pywinpty` env marker dep).
- PTY: backend in `pty_backend.py`; structured status (`alive`/`exit_code`/`data`); `read_pty(..., until=)` (**CSI sanitized** so host TTY is never reset); keys via `write_pty_keys` only (`\xHH`, `ctrl+x`, `key=esc`); secrets via `ask_into_pty` (not `write_pty` data); session limit/idle TTL/output budget; close terminate→kill.
- CLI limit hooks: interactive confirm to raise tool-loop rounds, PTY session cap, or PTY output budget. The prompt accepts `y`/empty (once), `n`, or `yyy` = stop asking for the rest of the turn (`cli.limits` module flag, reset by `run_turn_with_retries`, `/btw`, session switches, and chat exit). `--auto-continue` / `[agent] auto_continue_limits` set the process-wide default (covers one-shot/PTY); the flag does not imply YOLO and vice versa.
- Destructive confirms: `classify_danger` + `ToolRegistry(on_confirm=…)`; CLI default deny; config `confirm_destructive` / `path_denylist`. Soft confirm also for shells/REPLs and `python|bash -c` one-liners (argv + code preview); deny may include a user comment for the model. Session YOLO: `/yolo on|off|once` and `--yes` (skip soft confirms; hard denylists unchanged; `once` expires after the next user turn).
- **`vcs`**: read-only VCS tools (`vcs_kind` / `vcs_status` / `vcs_diff` / `vcs_log` / `vcs_branch`) via `VcsBackend` protocol; **git** implemented; detectors are pluggable for other systems.
- **`chat`**: human prompts as tools — `ask_user_line` / `ask_user_choice` / `ask_user_form` (shared `prompting` core); `wait` (line prompt with timeout; Enter disturbs). Container args are shape-checked (`tools/chat/shape.py`): a mis-typed string `options`/`fields` (or a non-string `question`) returns an explained error — `options="a, b"` used to iterate characters into a per-character menu. `wait`'s `duration` accepts numeric strings (`"5"` = 5s) and rejects booleans/text/containers.
- **`todo`**: LIFO stack of **task groups** (not a queue of tasks) — `todo_push` creates one group of siblings; `todo_pop` removes the whole top group; `todo_update` by id; DFS breakdown push[T1,T2]→push[T1.1…]→pop→push[T2.1…]; stored on session row; open items = unfinished work (`[TODO OPEN WORK]`); all-terminal non-empty = hygiene (`[TODO HYGIENE]`); nag channel via `[agent] todo_nag_strategy` (`developer`/`user` = prose nag; `synthetic_tool` = forged `todo_list` call + **real** `stack.render()` result + ToolCall/Result events; `none`).
- **`net` / `fetch`**: HTTP GET/POST/PUT/DELETE via niquests (isolated session, not the LLM client). Manual redirects with per-hop SSRF checks (`asyncio` `getaddrinfo`). Private/loopback hosts need instance policy grant (CLI timed confirm; **not** YOLO). Public cleartext HTTP and mutating methods soft-confirm (`YOLO`/`TRUSTABLE`). HTTPS→HTTP redirects blocked unless `allow_http_downgrade`. `user_agent` tool arg (or headers) is never replaced by the default when provided. Caps: body bytes/chars, request body size.
- **`mcp`** (`tools/mcp.py`): register one server's advertised tools into the catalog as namespaced `mcp__<server>__<tool>` definitions with `ToolSource(kind="mcp", plugin_id=server)`; tags LOCAL (+ READ_ONLY when the server config sets `read_only`) — never YOLO/TRUSTABLE. `register_mcp_tools` purges prior MCP entries then re-registers every connected server (hosts call it on tool-registry rebuild); connections must already be started by `McpManager`.
- **`skills`** (`tools/skills.py`): discovery, reading, and writing of skill directories (`plyngent.skills` owns roots, the `SKILL.md` frontmatter subset, the mtime-cached store, and the inside-the-skill path rule). `skill_list` (names, source root, directory, description, shadowed copies, frontmatter notes), `skill_read` (a `SKILL.md` by name, or one bundled file — reuses `read_file` for ranges/line numbers/`truncate_token`), `skill_search` (regex over every visible skill via `grep_files`, hits rebased on `<skill>/<file>`), `skill_create` (new `SKILL.md` + optional `files`), `skill_edit` (replace / `append` / rewrite one file; a rewritten `SKILL.md` is re-parsed and notes a lost description). Reads need no grant (the host publishes the roots as `WorkspacePolicy.static_read`, so `read_file`/`grep_files` see them while writes stay denied); writes go through **`SkillWritePolicy`** on the instance — denylist first, then `SessionState.skill_write_roots` (a `session` answer), then `[skills].allow_write`, then the host's confirm hook (CLI `prompt_skill_write_confirm`: `o`nce/`s`ession/`n`o, timed, default deny, **not** skipped by YOLO); a host with no hook denies. Grants are per **skill directory**; `[skills].enabled = false` drops the tools from the registry (`SKILL_TOOL_NAMES`).
- **`DEFAULT_TOOLS` / catalog**: access + file + process + vcs + chat + todo + skills + net (`fetch`) for local surface. `ToolTag.READ_ONLY` marks non-mutating tools (`read_file`, `get_truncated`, `listdir`, `glob_paths`, `grep_files`, `tree`, `todo_list`, `wait`, `ask_user_*`, `vcs_*`, `skill_list`/`skill_read`/`skill_search`, `fetch`); `ToolRegistry.clone(read_only_only=True)` selects just those for safe side turns / planning, and `read_only_context()` (in `tools/context.py`) makes `fetch` accept GET only.

### Prompting (`prompting.py`)

Shared interactive I/O: `ask` / `choose` / `form` / `confirm` with pluggable backend; non-TTY uses defaults or errors. CLI limit/confirm hooks and chat tools both use this. Async helpers serialize prompts (`run_prompt_async`).

### CLI (`cli/`)

Click app + readline REPL. Entry: `plyngent` / `python -m plyngent`.

- **`plyngent chat`**: provider/model (flags or interactive; Tab via readline in `prompting`); sessions store `provider_name`/`model` and restore on resume; SQLite via `[database]` (file DB under user data when url unset; explicit `url = ":memory:"` kept + warn); workspace-bound; resume latest for cwd/`--workspace` by default (`--new` / `--session`). One-shot: `-p/--prompt` and non-TTY stdin; exit codes 0/1/2/3; `--yes` (YOLO on), `--auto-continue` (auto-raise limits), `--stream/--no-stream`, `--quiet`. Root `--log-level`. Enabled `[mcp]` servers start before the REPL (`_start_mcp_manager`, per-server failures warn on stderr) and are closed on exit (`McpManager.aclose`); each server spawns in its own session/process group (`start_new_session` / `CREATE_NEW_PROCESS_GROUP`), so a terminal Ctrl+C does not kill it.
- Slash: Click group in `cli/slash.py` + `awaitlet` for async work; Tab completer from registry + ParamType `shell_complete`. `/history` is turn-oriented (`cli/transcript.py`): a turn runs from a user row to the next one, rows carry **display ordinals** (never DB row numbers; restart after `/clear`), the turn's ends (user + final answer) always print in full, `-v` expands every row (one-liners), `-vv` prints full bodies + local rows, `--preview` collapses every row, and `N`/`last [N]` address turns while `--message N` addresses one row. Multiline `"""` … `"""`; `/edit` via `$VISUAL`/`$EDITOR` (blocking only). `/yolo on|off|once` for soft destructive confirms. `/model --persist` / `/models --persist` write model catalog entries into TOML. `/todos` for human show/push/pop/clear of the todo stack. `/skills` (`list` default) prints the discovered skills with their roots, shadowed copies, and frontmatter notes; `/skills search PATTERN` and `/skills read NAME [--file F]` run the matching tool handlers (the slash path binds the tool context itself); `/skills reload` rescans the roots and rebuilds tools. `/mcp` (`list` default) prints per-server status/tool counts + an `instructions` preview when the server sent usage guidance, + last stderr lines; `/mcp reconnect` re-reads config, restarts every enabled server, and rebuilds the tool registry. `/grants` lists directory-access grants (`revoke <index|all>`; config grants point at the TOML); `/status` includes the grant count and the visible skill count (or `off` when `[skills].enabled` is false).
- `/btw` side questions: forked session; main transcript/DB untouched. `--tools=read` (default) exposes only `READ_ONLY` tools under `read_only_context()` (fetch=GET) with the main session view; `--tools=no` off; `--tools=full` whole-registry clone with fresh session. Tools off: default falls back to no tools; explicit `read`/`full` errors. The model gets an `aside` notice as its last message before the question (unsaved exchange + tool scope), shown to the user as `[notice] aside: …`.
- Explicit `/resume` or `--session` from another workspace prompts: **keep** / **update** / **abort**. A session resumed by a **freshly started process** (`--session` / default latest) also gets a `resume` notice about what a restart drops (PTY sessions, running commands, temp workspaces) — resuming inside a live process pushes nothing.
- Host notices (`agent/notices.py`): system-level messages appended as trailing `developer` rows, never system-prompt edits — `Notice.to_message()` for the model, `.summary` for the user via `cli.display.echo_notice` (`--quiet` hides the user line only). Durable hosts persist them (`ChatAgent.push_notice`); a side agent without memory keeps its notice local. Trailing developer messages never end a turn, so `/retry` still sees an interrupted turn behind a notice or checkpoint.
- Failed/cancelled turns: user kept; **committed tool rounds kept** (side effects not re-run on `/retry`); only unfinished assistant rolled back; Ctrl+C cancels the in-flight turn (a session-scoped asyncio SIGINT handler routes it, `cli/interrupt.py` — a stray Ctrl+C while the loop waits never tears the REPL down); TTY confirms/ask prompts run off-loop and register their own SIGINT target, so Ctrl+C there cancels just the poll-based read (`wait` reports `cancelled by user`) or is ignored with a hint for readline prompts (`ask_user_*`, confirms) — never the turn, never the loop; auto-retry waits 5s/10s/15s/20s then +10s (10 attempts), its countdown is Ctrl+C-cancellable, and the budget **resets when an attempt completes a model round** (connection recovered); then `/retry`.
- Pretty tool lines: every builtin tool prints its `* Verb 'target' ` prefix as the call starts and appends the outcome (`L1-4 (done)`, `(exit code 0)`, `(session 3)`) when the result lands, so a blocking tool stays visible while it runs; namespaced MCP tools share one generic line (`* MCP <server>:<tool> (done)`). Plugin tools and custom tool calls keep the `[tool] …` / `[tool ok] …` whole-line style. Interactive TTY only (`interactive_terminal()`): piped/plain output and `--verbose` keep whole lines. A second call before the first result (parallel batch) erases the open prefix — only that line, never the streamed line above it — and falls back to whole lines.
- **`plyngent providers`**, **`config path|edit`**. Config open: `$VISUAL`/`$EDITOR` (wait), else system default (`xdg-open` / `open` / `startfile`, non-blocking). `/edit` stays blocking-only. No providers → optional edit then reload when waited.
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
  - G6: README, `doc/plyngent.example.toml`, AGENTS overview/CLI notes

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
