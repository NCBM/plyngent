# plyngent

Single-user LLM chat and agent toolkit for the terminal.

Python **3.14+**. OpenAI-compatible APIs (including DeepSeek OpenAI-compat), OpenAI Responses with optional hosted tools, SQLite session memory, workspace-scoped file/process/VCS tools, and a readline REPL with slash commands.

Requires **Python 3.14+** on your `PATH` (or via [uv](https://docs.astral.sh/uv/) / [pipx](https://pipx.pypa.io/)).

## Install

### Quick try (`uvx`)

No permanent install — runs the published package in a temporary environment:

```bash
uvx plyngent --help
uvx plyngent chat
```

### User tool install

Keep `plyngent` on your PATH as a managed tool:

```bash
# uv (recommended)
uv tool install plyngent
plyngent --help

# pipx
pipx install plyngent
plyngent --help
```

Upgrade later:

```bash
uv tool upgrade plyngent
# or: pipx upgrade plyngent
```

### pip (venv or user)

```bash
python3.14 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install plyngent
plyngent --help
```

Or user install (if you accept that layout):

```bash
pip install --user plyngent
```

### From a git clone (development)

```bash
pdm install          # first time
pdm sync             # after pull
pdm run plyngent --help
```

Dev checks (same order as CI):

```bash
pdm run ruff check .
pdm run ruff format --check .   # or: pdm run ruff format .  to apply
pdm run basedpyright .
pdm run pytest
```

**Commit gateway** ([prek](https://prek.j178.dev/)): runs ruff check + format and basedpyright on `git commit` so format is not forgotten.

```bash
uv tool install prek    # once
prek install            # once per clone (installs .git/hooks/pre-commit)
prek run --all-files    # run all hooks on demand
```

Config: `prek.toml`. CI still runs the same checks in GitHub Actions.

## Basic usage

```bash
# 1) Create / open config
plyngent config path
plyngent config edit    # $VISUAL/$EDITOR, else system open (xdg-open/open/startfile)

# Minimal provider (OpenAI platform — Responses API; preset defaults to openai):
# [providers.oai]
# access_key_or_token = "sk-..."
# # models default: gpt-5.4, gpt-5.4-mini, gpt-5.4-nano
# # provider_tools default: web_search  (use provider_tools = [] to disable)

# 2) Chat
plyngent chat
plyngent chat --provider oai --model gpt-5.4-mini
plyngent chat -p "Summarize this repo" --provider oai --model gpt-5.4-mini --no-stream

# 3) List providers from config
plyngent providers

# 4) Plugins (entry-point allowlist under [plugins])
plyngent plugins list
plyngent plugins enable acme
```

In the REPL: type normally, use `/help` for slash commands, `"""` … `"""` for multiline, `/markdown` for Rich rendering, `/quit` to leave.

## Configure

Default config path (platformdirs):

```bash
plyngent config path
plyngent config edit    # $VISUAL/$EDITOR (e.g. codium --wait), else system default
```

Copy the example and fill in a real token:

```bash
cp doc/plyngent.example.toml "$(plyngent config path)"
# then edit providers
```

Minimal shape:

```toml
[providers.local]
preset = "openai-compatible"
url = "https://api.openai.com/v1"
access_key_or_token = "sk-..."
# Optional HTTP timeouts (seconds). Default: connect=10, read=600.
# timeout = 120
# timeout = { connect = 10, read = 600 }

[providers.local.models]
"gpt-4o-mini" = { text = true }

[agent]
# system_prompt = persona (omit → built-in). tool_directives = tool playbook.
# system_prompt = "" / tool_directives = "" disable each part; both "" → no system.
# Or multi-line override (prefer '''...'''):
# system_prompt = '''Your custom persona...'''
# tool_directives = '''### Workspace ...'''
confirm_destructive = true
max_context_tokens = 200000
# Fold MCP server initialize ``instructions`` (usage guidance) into the
# system prompt when tools are on (default true; false ignores server text).
# mcp_instructions = true
# Static out-of-workspace pre-allow (path → read|write|exec); never prompts.
# allow_paths = { "/data/datasets" = "read", "/tmp/build" = "exec" }
# Auto-raise tool/PTY limits without prompting (same as answering yyy).
# auto_continue_limits = false

# Optional plugins (entry-point names); default load none. See doc/plugins.md.
# [plugins]
# enable = ["acme"]
```

Per-provider **`timeout`** is passed to the HTTP session for chat/completions, Responses, and `GET /models`. A single number sets one timeout; `{ connect, read }` splits TCP/TLS setup vs idle wait between response bytes (SSE can run longer than `read` while chunks keep arriving). Tool/process timeouts (`run_argv`, PTY, policy confirm) are separate.

Third-party **plugins**: install a package that declares `project.entry-points."plyngent.tools"` (and later other groups), then allowlist the entry-point name under **`[plugins].enable`**. Details: [doc/plugins.md](doc/plugins.md).

**MCP servers** (Model Context Protocol over stdio): define servers under `[mcp.servers.<name>]` (`command` + `args`, optional `env`/`cwd`/`timeout`/`read_only`) and their tools become agent tools namespaced `mcp__<server>__<tool>` (LOCAL tags; `read_only = true` also marks them READ_ONLY, eligible for `/btw --tools=read`). Names in `[mcp].disable` stay disconnected. A server may also return optional `instructions` (usage guidance) in its MCP `initialize` response; when tools are on those are folded into the agent system prompt as `MCP server <name> instructions:` blocks (`[agent] mcp_instructions = false` disables), and `/mcp` previews them. In the REPL, `/mcp` lists per-server status and tool counts; `/mcp reconnect` re-reads the config file and restarts every enabled server (adopts newly added ones).

Supported provider presets today:

| Preset | API used by the agent | Notes |
|--------|------------------------|--------|
| `openai` (default if `preset` omitted) | OpenAI **Responses** (`POST /responses`) | Default models `gpt-5.4` / `gpt-5.4-mini` / `gpt-5.4-nano` when `models` is omitted; optional `provider_tools` (default `web_search`) |
| `openai-compatible` | Chat Completions | Generic hosts (vLLM, LiteLLM, proxies); requires `url` |
| `deepseek` | Chat Completions (DeepSeek) | Default models `deepseek-v4-flash` / `deepseek-v4-pro` when `models` is omitted; set `convention = "responses"` (provider or per-model) to use the OpenAI **Responses** API (`POST /responses`, currently only `deepseek-v4-flash`), or `convention = "anthropic"` to use the Anthropic **Messages** API on `https://api.deepseek.com/anthropic` |
| `anthropic` | Anthropic **Messages** (`POST /messages`) | Native tools/streaming; set `access_key_or_token` (API key) |

Model-level `preset` / `url` overrides on a catalog entry can route a single provider name to a different API (e.g. gateway + Anthropic model). DeepSeek models may also carry a per-model `convention` (empty = inherit the provider-level one), e.g. serve `deepseek-v4-flash` over Responses while `deepseek-v4-pro` stays Chat Completions:

```toml
[providers.deepseek]
preset = "deepseek"
access_key_or_token = "sk-..."

[providers.deepseek.models]
"deepseek-v4-flash" = { text = true, convention = "responses" }
"deepseek-v4-pro" = { text = true }
```

DeepSeek's Anthropic convention (`convention = "anthropic"`) points at
`https://api.deepseek.com/anthropic` by default and is fully supported by the
agent's Anthropic Messages path (tools, streaming, usage). DeepSeek maps
`claude-opus*` model ids to `deepseek-v4-pro` and `claude-haiku*` /
`claude-sonnet*` (and unknown ids) to `deepseek-v4-flash` server-side; real
`deepseek-v4-*` ids pass through as-is. `GET /models` is not documented on that
base, so model selection is config-driven.

If `[database]` is omitted (or SQLite `url` is unset/empty), chat uses a durable file under the user data dir (e.g. `~/.local/share/plyngent/chat.db` on Linux). Set `url = ":memory:"` for a true in-memory SQLite (CLI warns; no file; useful for tests).

## Chat

### Interactive REPL

```bash
plyngent chat
plyngent chat --provider local --model gpt-4o-mini
plyngent chat --workspace /path/to/project --new
plyngent chat --session 3
```

| Flag | Meaning |
|------|---------|
| `--provider` / `--model` | Select from config (required when multiple and non-interactive) |
| `--workspace` | Tool root (default: cwd); sessions bind to this path |
| `--new` / `--session ID` | Fresh session vs resume by id |
| `--tools` / `--no-tools` | Default tools on |
| `--max-rounds` | Tool-loop rounds per turn (default 32) |
| `--stream` / `--no-stream` | Streaming deltas (default on) |
| `--quiet` | Less status on stderr |
| `--yes` | YOLO on: skip destructive-tool confirms for this process |
| `--auto-continue` | Auto-raise tool/PTY limits without prompting (like answering `yyy`) |
| `--log-level` | On the root CLI: `DEBUG`, `INFO`, `WARNING`, … |

Sessions resume the **most recently updated** session for the current workspace unless you pass `--new` or `--session`. Each session remembers the last **provider** and **model** (restored on resume so you are not re-prompted). A session resumed by a fresh `plyngent` process also gets a `[notice] resume: …` line: the model is told that process-scoped state — PTY sessions, running commands, temporary workspaces — did not survive the restart, and that an unfinished turn may have been cut off. The notice is kept with the transcript; `--quiet` hides only the user-facing line.

When a tool-loop or PTY limit is hit, the prompt accepts `y` (continue once), `n` (stop), or `yyy` — **stop asking for the rest of this turn** (the next user turn prompts again). `--auto-continue` / `[agent] auto_continue_limits = true` skip the prompt for every turn.

Tool calls show progress as they run: known tools (`read_file`, `run_argv`, `grep_files`, …) print a `* Read 'src/x.py' ` prefix the moment the call starts, and the outcome (`L1-80 (done)`, `(exit code 0)`) is appended once the result lands. Piped output and `--verbose` print the finished line as before.

### One-shot (scripts / CI)

```bash
plyngent chat -p "Summarize README.md" --provider local --model gpt-4o-mini --no-stream
echo "hello" | plyngent chat --provider local --model gpt-4o-mini
```

Exit codes (one-shot):

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Config / usage error |
| 2 | Cancelled |
| 3 | Turn failed (API / incomplete) |

### Input ergonomics

- **Multiline**: start a message with `"""`, end a later line with `"""`.
- **`/edit`**: compose a turn in `$VISUAL`/`$EDITOR` (blocking only; empty cancels).
- **Tab**: completes slash commands and some arguments (provider, model, on/off, export, `/help` targets).

### Slash commands

Type `/help` in the REPL for the live list. Common ones:

| Command | Purpose |
|---------|---------|
| `/status` | Provider, session, context/usage estimates |
| `/history [N \| last [N]]` | Conversation grouped by turn: default = the last turn (your message + the model's answer); `N` = turn N, `last N` = the last N turns |
| `/history -v` / `-vv` | `-v` prints every row of the selected turn(s) (tool calls/results, notices); `-vv` prints full bodies (and the local system row). `--message N` prints one message in full, `--preview` one-lines every row. Numbers are display ordinals shown by `/history`, never database rows (they restart after `/clear`) |
| `/sessions` | Sessions for this workspace |
| `/new` `/resume` `/rename` `/delete` | Session lifecycle (`/delete` confirms) |
| `/export [md\|json] [path]` | Transcript from DB (no secrets) |
| `/compact` | Soft-compact + model summary into a **new** session |
| `/stream` `/verbose` `/markdown` `/tools` `/rounds` | Toggles and limits |
| `/yolo [on\|off\|once]` | Soft destructive confirms: sticky skip, off, or next turn only |

| `/retry` | Re-run incomplete last user turn (after error/cancel) |
| `/btw [--tools read\|no\|full] [--fresh]` | Side question without changing the main session (read-only tools by default; the model is told the exchange is unsaved and how its tools are scoped) |
| `/provider` `/model` | Switch without restarting |
| `/model --persist` | Save current model id into `plyngent.toml` catalog |
| `/models` | List config + remote `GET /models` (always re-fetches) |
| `/models --persist` | Merge remote catalog into TOML for this provider |
| `/todos` | Todo/task stack: list, push, pop, done, clear |
| `/grants` | Directory-access grants: list, or `revoke <index\|all>` |
| `/config` | Edit `plyngent.toml` ($VISUAL/$EDITOR or system open); reload after blocking editor |
| `/quit` | Leave the REPL |

User messages are saved immediately. On API error or Ctrl+C, partial assistant/tool output is discarded but the user message stays so `/retry` works after resume. Interactive auto-retry waits 5s, 10s, 15s, 20s, then +10s each step (10 attempts); an attempt that gets a full model round back **resets that budget**, so a connection that recovers and drops again starts over from 5s.

Ctrl+C cancels the in-flight turn, and during the auto-retry countdown it cancels the retry. Model-initiated prompts run off the event loop: a `wait` prompt is cancelled by Ctrl+C (the tool reports `cancelled by user` and the turn continues), while `ask_user_*` and confirm prompts ignore it with a one-line hint — answer them to continue. Ctrl+C never exits the REPL (use Ctrl+D or `/quit`), and MCP servers run in their own process group, so it never kills them either.

## Workspace model

- **Workspace** = root for file/process/VCS tools (default cwd).
- **Session** = SQLite chat bound to a workspace path.
- Paths outside the workspace need a **directory-access grant**: the model calls `request_directory_access` (`read` / `write` / `exec`), the human approves at the requested level or another one (timed y/N, default deny; `--yes`/`/yolo` auto-approve as a process-only grant), and approval lasts for the session (`/grants` lists/revokes). Static pre-allow lives under `[agent].allow_paths`. The path denylist always wins, and grants gate path-resolving tools only — they are **not** a sandbox (`run_argv`/PTY reach the whole filesystem).
- Resuming a session from another directory prompts: keep session workspace, rebind to current, or abort.

## Tools (when enabled)

Default registry: file ops (including `tree` with a markdown bullet-list default, `flat` paths or classic `decorated` on demand; default noise-dir skips), `run_argv` / `run_argv_batch` / PTY (POSIX openpty; Windows ConPTY via pywinpty), read-only VCS (git), HTTP `fetch` (GET/POST/PUT/DELETE via niquests; private/loopback hosts need a human policy grant, not YOLO), human prompts (`ask_user_line` / `ask_user_choice` / `ask_user_form`), `wait` (line prompt with timeout; Enter disturbs), `request_directory_access` (out-of-workspace access grants), `get_truncated` (resume any truncated result via its `truncate_token=...`), and todo stack tools (`todo_list` / `todo_push` / `todo_pop` / `todo_update` / `todo_clear`).

Safety defaults:

- Paths stay under the workspace unless granted (`request_directory_access` / `[agent].allow_paths`); optional `path_denylist` substrings always apply (`tree` also skips denylisted children by default).
- Command basename denylist (e.g. dangerous shells/utilities).
- Destructive tools (delete/move/overwrite) can require confirm (`confirm_destructive`; default deny in non-TTY). Override for the session with `/yolo on|off|once` or startup `--yes` (path/command denylists still apply).
- PTY sessions: caps, idle TTL, output budget; master FD is non-inheritable; sessions closed on chat exit.
  Prefer file tools over full-screen editors (`vim`/`nano`) for edits. `read_pty` sanitizes CSI/controls
  so tool results cannot reprogram the host TTY (no host terminal reset on exit).
  `write_pty` is literal text only; use `write_pty_keys` for `\xHH`, `ctrl+x`, `key=esc|enter|…`.
  For passwords/sudo/ssh prompts use `ask_into_pty` (human types locally; answer never returns to the model).

- Mis-typed tool arguments are answered, never guessed: a string where an array is expected (`run_argv` `argv`, `ask_user_choice` `options`, `ask_user_form` `fields`), a call that omits a required argument, or a non-numeric `wait` duration (numeric strings are accepted) returns an error showing the shape/names to send — the human never sees a garbage prompt, and no confirm prompt fires for a call that cannot run.

## Usage / context (CLI)

- **Context size** prefers API `prompt_tokens` from the last model call; otherwise a char-based estimate (~4 chars/token).
- **Turn/session usage** sum billed completion usage across tool rounds (history is re-sent each round).
- Soft compact can calibrate from reported `prompt_tokens`. See `/status`.

## Other commands

```bash
plyngent providers          # list configured providers
plyngent config path|edit
plyngent --log-level INFO chat ...
```

## Architecture (short)

See [doc/architecture.md](doc/architecture.md) and [AGENTS.md](AGENTS.md) for developers.

- **`lmproto/`** — OpenAI-compatible, OpenAI Responses, Anthropic Messages, DeepSeek msgspec models and async SSE clients  
- **`agent/`** — kind-based tool loop (`chat_completions` / `responses` / `messages`), streaming, usage, compact  
- **`memory/`** — async SQLAlchemy sessions/messages  
- **`tools/`** — workspace tools  
- **`cli/`** — Click entry + slash registry (`awaitlet` bridges sync Click to async work)  
- Multi-tenant / web (`router/`, real `web/`) are **not** in scope for the single-user CLI (Phase H).

## License

MIT
