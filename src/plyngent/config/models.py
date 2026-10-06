from typing import Any, Literal

from msgspec import Struct, field

# DeepSeek API surface convention (``DeepseekProvider.convention`` and
# per-model ``ModelConfig.convention`` override).
# - "" (default) / "openai" → OpenAI-compatible chat completions
# - "responses"           → OpenAI Responses API (``POST /responses``)
# - "anthropic"           → Anthropic Messages API (``POST /messages`` on
#                           ``https://api.deepseek.com/anthropic``)
type DeepSeekConvention = Literal["", "openai", "anthropic", "responses"]

# Static directory pre-allow mode for ``[agent].allow_paths`` (path → mode).
type AllowPathMode = Literal["read", "write", "exec"]

# Thinking strength, normalized across API surfaces (``reasoning_effort`` on
# ``[agent]``, a provider, or a single model; the most specific one wins).
# "" = send nothing and let the provider pick its own default.
# Mapped per surface by ``config.reasoning``: chat completions / Responses send
# ``reasoning_effort`` / ``reasoning.effort`` (``max`` is DeepSeek-only and
# clamps to ``xhigh`` elsewhere), Anthropic maps it onto ``thinking``.
type ReasoningEffortConfig = Literal["", "none", "minimal", "low", "medium", "high", "xhigh", "max"]

# Built-in skill discovery sources (``[skills].discover``): our own user config
# directory, and the skill directories other harnesses install into.
type SkillDiscoverySource = Literal["plyngent", "claude"]

# Built-in persona when ``[agent].system_prompt`` is omitted.
# Set ``system_prompt = ""`` to omit the persona block only.
# Override with a multi-line TOML literal (prefer ''' so nested " is fine).
DEFAULT_SYSTEM_PROMPT = """\
You are a professional coding agent in a workspace-bound tool environment.
"""

# Built-in tool playbook when ``[agent].tool_directives`` is omitted.
# Set ``tool_directives = ""`` to omit this block only.
DEFAULT_TOOL_DIRECTIVES = """\
### Workspace
- Explore with `tree`/`listdir`/`glob_paths`/`regex_files`/`read_file` when unsure.
- Stay under the workspace (or a path from `new_temporary_workspace`). Respect path denylists.
- Need a path outside the workspace? Call `request_directory_access` \
(human approves; denylists apply) instead of retrying an escaped path.

### Files
- Prefer file tools over shell. `edit_replace` fails usually means a bad match — \
fix `old_string` or `max_replaces`; shell workarounds rarely help.
- `edit_replace` defaults to the first match; if remaining matches are reported, \
raise `max_replaces` or narrow `old_string`.
- `edit_lineno` edits only lines you read via `read_file(path, with_lineno=true)`; \
re-read with line numbers after any `edit_*` / `write_file` / copy/move/delete, \
since line numbers go stale.
- `read_file` results start with a 1-based line range `L{begin}-{end}` (`offset` \
is 0-based); `with_lineno` shows per-line numbers instead.
- `regex_files` takes a Python `re` regex for `pattern`, not a filename glob \
(`*.py` / `**/*.py` fail with `invalid regex`), and a literal file/dir `path` \
(no glob expansion, so `src/**/*.py` fails with `path does not exist`); there is \
no file-type filter, so narrow with `path` and filter hits yourself.
- Truncated results (any tool) carry a `truncate_token=...`; use `get_truncated` \
with it to keep reading (chains through truncations).

### Commands
- Prefer `run_argv` / `run_argv_batch` (argv lists, no shell) over `bash -c` or similar.
- A shell or interpreter (`bash` / `python` / `node` / …) always requires the user to confirm — \
a `-c` one-liner, a script (`python x.py`), or a bare interactive shell alike — and each \
confirm blocks the turn, so it costs time. Invoke the target command directly (argv) instead \
of wrapping it in `bash -c` / `python -c` / `node`, and reach for a dedicated tool \
(`read_file` / `edit_*` / `vcs_*` / `fetch` / …) when one already does the job.
- Several `run_argv` calls in one step may run in parallel; use `run_argv_batch` \
for ordered pipelines (`pipe_out` / `mix_stderr` as needed).
- Prefer `vcs_*` for status/diff/log/branch when enough.

### Network
- Prefer `fetch` (GET/POST/PUT/DELETE) for HTTP(S) docs/APIs over curl/wget via shell.
- Set `user_agent` (or a User-Agent header) when the remote expects a specific client; \
otherwise a small default is used. Never rely on shell to spoof identity.
- Private/loopback/LAN hosts need an explicit human policy allow (not skipped by YOLO).
- Prefer provider hosted search (e.g. web_search) for open-ended research; use `fetch` for known URLs.
- Respect denials, size limits, and HTTPS→HTTP redirect blocks; do not re-fetch the same URL repeatedly.

### Humans & safety
- Prefer `ask_user_line` / `ask_user_choice` / `ask_user_form` over waiting for the next free-form turn.
- Use `wait` to pause; pressing Enter during the wait disturbs it (reason optional; turn continues immediately).
- Overwrites, deletes, shells, and risky ops may require human confirm; hard denylists are not skipped by YOLO.
- Temporary scratch: `new_temporary_workspace`.

### PTY
- Use PTY for interactive/TUI, sudo, ssh, or live servers.
- Passwords and other secret input: only `ask_into_pty` (never echo secrets into `write_pty`).
- Control keys: `write_pty_keys`, not literal `write_pty` data.

### Todos
- Use the todo stack for multi-step work (LIFO groups: push related items, finish/update, \
pop the group). Open items mean unfinished work.
- The stack does **not** auto-clear. When every task is done (all items done/cancelled), \
call `todo_clear` (or pop finished TOP groups) so no hygiene noise is left behind.
"""


def compose_agent_system_content(*parts: str) -> str | None:
    """Join persona + tool playbook + extra blocks into one system body.

    Returns ``None`` when every part is empty. Non-empty parts are stripped and
    joined with a blank line. Any part may be ``""`` to disable that block
    (e.g. ``system_prompt = ""`` keeps the tool playbook).
    """
    joined = "\n\n".join(part.strip() for part in parts if part and part.strip())
    return joined or None


class DatabaseConfig(Struct, omit_defaults=True):
    """Database connection configuration.

    ``url`` is ``None`` when unset. The CLI fills a durable user-data
    ``chat.db`` only for unset/empty url. Explicit ``":memory:"`` is a true
    in-memory SQLite (no rewrite).
    """

    implementation: str = "sqlite"
    url: str | None = None
    username: str | None = None
    password: str | None = None


class AgentConfig(Struct, omit_defaults=True):
    """Single-user agent profile defaults."""

    # Persona / role (omit → DEFAULT_SYSTEM_PROMPT; "" disables persona only).
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    # Tool how-to (omit → DEFAULT_TOOL_DIRECTIVES; "" disables playbook only).
    tool_directives: str = DEFAULT_TOOL_DIRECTIVES
    # Fold connected MCP servers' initialize ``instructions`` (usage guidance)
    # into the agent system prompt when tools are on. false = ignore server text.
    mcp_instructions: bool = True
    max_tool_result_chars: int = 32_000
    parallel_tools: bool = True
    confirm_destructive: bool = True
    path_denylist: list[str] = field(default_factory=list)
    # Static directory pre-allow (resolved path → mode); never prompts.
    # Denylist still applies, and the path must exist when chat starts.
    allow_paths: dict[str, AllowPathMode] = field(default_factory=dict)
    # Auto-raise tool/PTY limits without prompting (same as answering ``yyy``);
    # process-wide, so every turn skips the limit confirm.
    auto_continue_limits: bool = False
    max_context_tokens: int = 200_000

    # How to inject todo stack nags into model context (see agent/todo_nag.py).
    # developer | user | synthetic_tool | none  (legacy "system" → developer)
    todo_nag_strategy: str = "developer"

    # Append-only developer playbook checkpoints when last prompt_tokens crosses
    # N * this interval (0 = off). See agent/directive_checkpoint.py.
    directive_reminder_tokens: int = 100_000
    # Optional short checklist body (empty = built-in hard-constraint list).
    directive_reminder_text: str = ""

    # Compact / summarisation prompts (empty = use built-in defaults).
    compact_system_prompt: str = ""
    compact_user_prefix: str = ""
    compact_seed_text: str = ""

    # Thinking strength for every request (a provider or single model may
    # override either value; the most specific one wins).
    # reasoning_effort: "" | none | minimal | low | medium | high | xhigh | max.
    # thinking_budget_tokens: explicit Anthropic ``thinking`` budget (0 = derive
    # it from ``reasoning_effort``). See config/reasoning.py.
    reasoning_effort: ReasoningEffortConfig = ""
    thinking_budget_tokens: int = 0


class PluginsConfig(Struct, omit_defaults=True):
    """Third-party plugins (not tool-specific config).

    Hosts allowlist plugin **entry-point names** (today: group ``plyngent.tools``;
    other extension points may reuse the same allowlist later).

    - ``enable`` empty / omitted → load **no** plugins (safe default).
    - ``enable = ["*"]`` → load every discovered entry point for that group.
    - ``disable`` always wins over ``enable`` / ``*``.
    """

    enable: list[str] = field(default_factory=list)
    disable: list[str] = field(default_factory=list)


class McpServerConfig(Struct, omit_defaults=True):
    """One MCP server definition (stdio transport today).

    TOML example::

        [mcp.servers.docs]
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env = { RUST_LOG = "info" }
        # cwd = "/path"            # spawn working directory (empty = inherit)
        # timeout = 30.0            # per-request timeout in seconds
        # read_only = true          # mark every tool READ_ONLY (safe side turns)

    ``url`` is reserved for the streamable-HTTP transport; empty = stdio.
    A defined server is connected unless its name appears in ``[mcp].disable``.
    """

    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout: float = 30.0
    read_only: bool = False
    url: str = ""


class McpConfig(Struct, omit_defaults=True):
    """Model Context Protocol servers exposed as agent tools.

    Defining a server under ``[mcp.servers.<name>]`` is the opt-in; tools are
    namespaced ``mcp__<server>__<tool>`` and registered with LOCAL tags. Names
    in ``disable`` stay disconnected even when defined.
    """

    servers: dict[str, McpServerConfig] = field(default_factory=dict)
    disable: list[str] = field(default_factory=list)


class NetworkingConfig(Struct, omit_defaults=True):
    """Host-side network policy for tools such as ``fetch``.

    ``ssrf_assume_public_cidrs``: CIDR strings whose resolved addresses are treated
    as **public for SSRF policy** (skip private-host grants). Use for Clash/meta
    Fake-IP pools (commonly ``198.18.0.0/15``). Empty default = no exemptions.

    Metadata ranges (e.g. ``169.254.169.254``) stay forbidden even if listed here.
    Real loopback/RFC1918 literals are unaffected unless you put those CIDRs here
    (not recommended).
    """

    ssrf_assume_public_cidrs: list[str] = field(default_factory=list)


class SkillsConfig(Struct, omit_defaults=True):
    """Skill directories the agent may discover, read, and (with a grant) edit.

    A skill is a directory holding a ``SKILL.md`` (YAML frontmatter with at least
    ``name`` / ``description``, then a Markdown body) plus optional bundled files
    (scripts, references, templates). Roots are scanned in this order:

    1. ``paths`` — explicit roots, highest priority
    2. ``discover`` sources in order: ``plyngent`` = our user config dir
       (``<user config>/skills``), ``claude`` = Claude Code's skill dirs
       (``~/.claude/skills``; plus ``<workspace>/.claude/skills`` when
       ``include_project_roots``)

    A skill found in several roots shadows the lower-priority copies: they stay
    listed, never hidden. ``enabled = false`` registers no skill tools at all and
    injects no catalog. Writing needs an explicit grant (a confirm per call, or
    ``allow_write`` for a standing one); reads need none.
    """

    enabled: bool = True
    paths: list[str] = field(default_factory=list)
    discover: list[SkillDiscoverySource] = field(default_factory=lambda: ["plyngent", "claude"])
    include_project_roots: bool = True
    inject_catalog: bool = True
    max_catalog_skills: int = 50
    allow_write: bool = False


class ModelConfig(Struct, omit_defaults=True):
    """Capability flags and optional routing overrides for a model.

    ``preset`` / ``url`` override the parent provider for this model only.
    ``preset`` keeps the same meanings as provider presets:
    ``openai`` → /responses, ``openai-compatible`` → /chat/completions,
    ``anthropic`` → /messages, ``deepseek`` → DeepSeek.
    """

    text: bool = True
    image_in: bool = False
    image_out: bool = False
    audio_in: bool = False
    audio_out: bool = False
    video_in: bool = False
    video_out: bool = False
    cost_factor: float = 1.0
    preset: str = ""
    url: str = ""
    # DeepSeek API surface override ("" = inherit provider convention).
    convention: DeepSeekConvention = ""
    # Thinking strength for this model only ("" / 0 = inherit the provider).
    reasoning_effort: ReasoningEffortConfig = ""
    thinking_budget_tokens: int = 0


class HttpTimeoutConfig(Struct, omit_defaults=True):
    """Per-provider HTTP timeouts (seconds) for the LLM API client.

    TOML examples::

        timeout = 120
        timeout = { connect = 10, read = 600 }

    ``connect`` bounds TCP/TLS setup; ``read`` bounds idle wait between response
    bytes (SSE streams can run longer than *read* as long as chunks keep arriving).
    Omitted fields use product defaults (10s connect / 600s read).
    """

    connect: float | None = None
    read: float | None = None


class ProviderConfig(Struct, tag_field="preset", omit_defaults=True):
    """Base config for all LLM providers.

    The ``preset`` tag field acts as the tagged-union discriminator
    and is auto-generated by msgspec — do not re-declare it in subclasses.
    """

    access_key_or_token: str
    url: str = ""
    models: dict[str, ModelConfig] = field(default_factory=dict)
    # float = single timeout for the session; table = connect/read split; omit = defaults.
    timeout: float | HttpTimeoutConfig | None = None
    # Thinking strength for every model of this provider ("" / 0 = inherit [agent]).
    reasoning_effort: ReasoningEffortConfig = ""
    thinking_budget_tokens: int = 0


def _default_openai_models() -> dict[str, ModelConfig]:
    """Current OpenAI text catalog when TOML omits ``models``."""
    return {
        "gpt-5.4": ModelConfig(text=True),
        "gpt-5.4-mini": ModelConfig(text=True),
        "gpt-5.4-nano": ModelConfig(text=True),
    }


def _default_openai_provider_tools() -> list[dict[str, Any]]:
    """Hosted tools when TOML omits ``provider_tools``."""
    return [{"type": "web_search"}]


class OpenAIProvider(ProviderConfig, tag="openai"):
    """OpenAI platform provider (agent uses Responses API).

    ``url`` should include the API version path (default: ``https://api.openai.com/v1``).
    ``provider_tools`` are hosted/provider-side tools (e.g. web_search) passed
    through to ``POST /responses`` as opaque dicts. They are **not** local
    ``ToolRegistry`` handlers and are ignored by non-OpenAI clients.

    When ``models`` is omitted in TOML, seeds a small default catalog (same idea
    as DeepSeek). Explicit ``models = {}`` stays empty (recoverable / free-form).
    When ``provider_tools`` is omitted, defaults to web_search; use
    ``provider_tools = []`` to disable hosted tools.
    """

    models: dict[str, ModelConfig] = field(default_factory=_default_openai_models)
    provider_tools: list[dict[str, Any]] = field(default_factory=_default_openai_provider_tools)


class OpenAICompatibleProvider(ProviderConfig, tag="openai-compatible"):
    """Generic OpenAI-compatible provider (e.g. vLLM, LiteLLM)."""


class AnthropicProvider(ProviderConfig, tag="anthropic"):
    """Anthropic API provider."""


def _default_deepseek_models() -> dict[str, ModelConfig]:
    """Current DeepSeek text catalog when TOML omits ``models``."""
    return {
        "deepseek-flash": ModelConfig(text=True),
        "deepseek-v4-pro": ModelConfig(text=True),
    }


class DeepseekProvider(ProviderConfig, tag="deepseek"):
    """DeepSeek API provider.

    ``convention`` selects the API surface on DeepSeek's endpoints:
    - ``""`` / ``"openai"`` → OpenAI-compatible chat completions (default)
    - ``"responses"`` → OpenAI Responses API (``POST /responses``)
    - ``"anthropic"`` → Anthropic Messages API (``POST /messages`` on
      ``https://api.deepseek.com/anthropic``)

    ``extras`` keeps arbitrary provider keys; the legacy ``extras.convention``
    key is parsed but ignored (prefer the typed ``convention`` field).
    """

    models: dict[str, ModelConfig] = field(default_factory=_default_deepseek_models)
    convention: DeepSeekConvention = ""
    extras: dict[str, str] = field(default_factory=dict)


type Provider = OpenAIProvider | OpenAICompatibleProvider | AnthropicProvider | DeepseekProvider
