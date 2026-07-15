# plyngent

Single-user LLM chat and agent toolkit for the terminal.

Python **3.14+**, managed with [PDM](https://pdm-project.org/). OpenAI-compatible APIs (including DeepSeek OpenAI-compat), SQLite session memory, workspace-scoped file/process/VCS tools, and a readline REPL with slash commands.

## Install

```bash
# from a clone
pdm install
pdm run plyngent --help

# or editable install into your environment
pdm install
# entry point: plyngent
```

Dev checks:

```bash
pdm run basedpyright .
pdm run ruff check .
pdm run ruff format .
pdm run pytest
```

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

If `[database]` is omitted (or SQLite `url` is empty/`":memory:"`), chat uses a durable file under the user data dir (e.g. `~/.local/share/plyngent/chat.db` on Linux).

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
| `--yes` | Allow destructive tools without confirm (also for one-shot) |
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
| `/history [n]` | Recent messages |
| `/sessions` | Sessions for this workspace |
| `/new` `/resume` `/rename` `/delete` | Session lifecycle (`/delete` confirms) |
| `/export [md\|json] [path]` | Transcript from DB (no secrets) |
| `/compact` | Soft-compact + model summary into a **new** session |
| `/stream` `/verbose` `/tools` `/rounds` | Toggles and limits |
| `/retry` | Re-run incomplete last user turn (after error/cancel) |
| `/provider` `/model` | Switch without restarting |
| `/models` | List config + remote `GET /models` (`--refresh` bypasses cache) |
| `/config` | Edit `plyngent.toml` in `$EDITOR` and reload |
| `/quit` | Leave the REPL |

User messages are saved immediately. On API error or Ctrl+C, partial assistant/tool output is discarded but the user message stays so `/retry` works after resume. Interactive auto-retry uses 10s / 20s / 30s delays.

## Workspace model

- **Workspace** = root for file/process/VCS tools (default cwd).
- **Session** = SQLite chat bound to a workspace path.
- Resuming a session from another directory prompts: keep session workspace, rebind to current, or abort.

## Tools (when enabled)

Default registry: file ops, `run_command` / PTY (POSIX openpty; Windows ConPTY via pywinpty), read-only VCS (git), and human prompts (`ask_user` / `choose_user` / `form_user`).

Safety defaults:

- Paths stay under the workspace; optional `path_denylist` substrings.
- Command basename denylist (e.g. dangerous shells/utilities).
- Destructive tools (delete/move/overwrite) can require confirm (`confirm_destructive`; default deny in non-TTY).
- PTY sessions: caps, idle TTL, output budget; master FD is non-inheritable; sessions closed on chat exit.

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
