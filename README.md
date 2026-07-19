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
plyngent config edit    # needs $EDITOR

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
```

In the REPL: type normally, use `/help` for slash commands, `"""` … `"""` for multiline, `/markdown` for Rich rendering, `/quit` to leave.

## Configure

Default config path (platformdirs):

```bash
plyngent config path
plyngent config edit    # opens $EDITOR (e.g. codium --wait)
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

[providers.local.models]
"gpt-4o-mini" = { text = true }

[agent]
system_prompt = "You are a careful coding assistant."
confirm_destructive = true
max_context_tokens = 200000
```

Supported provider presets today: `openai` (default if `preset` is omitted; default models `gpt-5.4` / `gpt-5.4-mini` / `gpt-5.4-nano` when `models` is omitted), `openai-compatible`, `deepseek` (OpenAI convention; default models `deepseek-v4-flash` and `deepseek-v4-pro` if `models` is omitted). Anthropic presets are modeled in config but not wired in the runtime client yet.

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
| `--log-level` | On the root CLI: `DEBUG`, `INFO`, `WARNING`, … |

Sessions resume the **most recently updated** session for the current workspace unless you pass `--new` or `--session`. Each session remembers the last **provider** and **model** (restored on resume so you are not re-prompted).

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
- **`/edit`**: compose a turn in `$EDITOR` (empty buffer cancels).
- **Tab**: completes slash commands and some arguments (provider, model, on/off, export, `/help` targets).

### Slash commands

Type `/help` in the REPL for the live list. Common ones:

| Command | Purpose |
|---------|---------|
| `/status` | Provider, session, context/usage estimates |
| `/history [n\|last]` | Recent messages (default preview; `last`/`1` = full + markdown) |
| `/history --full` | Full bodies for the selected window |
| `/sessions` | Sessions for this workspace |
| `/new` `/resume` `/rename` `/delete` | Session lifecycle (`/delete` confirms) |
| `/export [md\|json] [path]` | Transcript from DB (no secrets) |
| `/compact` | Soft-compact + model summary into a **new** session |
| `/stream` `/verbose` `/markdown` `/tools` `/rounds` | Toggles and limits |
| `/yolo [on\|off\|once]` | Soft destructive confirms: sticky skip, off, or next turn only |

| `/retry` | Re-run incomplete last user turn (after error/cancel) |
| `/provider` `/model` | Switch without restarting |
| `/model --persist` | Save current model id into `plyngent.toml` catalog |
| `/models` | List config + remote `GET /models` (always re-fetches) |
| `/models --persist` | Merge remote catalog into TOML for this provider |
| `/todos` | Todo/task stack: list, push, pop, done, clear |
| `/config` | Edit `plyngent.toml` in `$EDITOR` and reload |
| `/quit` | Leave the REPL |

User messages are saved immediately. On API error or Ctrl+C, partial assistant/tool output is discarded but the user message stays so `/retry` works after resume. Interactive auto-retry uses 10s / 20s / 30s delays.

## Workspace model

- **Workspace** = root for file/process/VCS tools (default cwd).
- **Session** = SQLite chat bound to a workspace path.
- Resuming a session from another directory prompts: keep session workspace, rebind to current, or abort.

## Tools (when enabled)

Default registry: file ops (including `tree` with default noise-dir skips), `run_command` / PTY (POSIX openpty; Windows ConPTY via pywinpty), read-only VCS (git), human prompts (`ask_user_line` / `ask_user_choice` / `ask_user_form`), and todo stack tools (`todo_list` / `todo_push` / `todo_pop` / `todo_update` / `todo_clear`).

Safety defaults:

- Paths stay under the workspace; optional `path_denylist` substrings (`tree` also skips denylisted children by default).
- Command basename denylist (e.g. dangerous shells/utilities).
- Destructive tools (delete/move/overwrite) can require confirm (`confirm_destructive`; default deny in non-TTY). Override for the session with `/yolo on|off|once` or startup `--yes` (path/command denylists still apply).
- PTY sessions: caps, idle TTL, output budget; master FD is non-inheritable; sessions closed on chat exit.
  Prefer file tools over full-screen editors (`vim`/`nano`) for edits. `read_pty` sanitizes CSI/controls
  so tool results cannot reprogram the host TTY (no host terminal reset on exit).
  `write_pty` is literal text only; use `write_pty_keys` for `\xHH`, `ctrl+x`, `key=esc|enter|…`.

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

See [doc/architecture.md](doc/architecture.md) and [CLAUDE.md](CLAUDE.md) for developers.

- **`lmproto/`** — OpenAI-compatible (+ DeepSeek) msgspec models and async SSE clients  
- **`agent/`** — tool loop, streaming, usage, compact  
- **`memory/`** — async SQLAlchemy sessions/messages  
- **`tools/`** — workspace tools  
- **`cli/`** — Click entry + slash registry (`awaitlet` bridges sync Click to async work)  
- Multi-tenant / web (`router/`, real `web/`) are **not** in scope for the single-user CLI (Phase H).

## License

MIT
