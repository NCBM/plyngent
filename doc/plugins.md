# Tool plugins

Third-party tools install as Python packages and register through the
**`plyngent.tools`** entry-point group. The host **allowlists** which plugins
load; default is **load none**.

Importing a plugin only **registers** tools into the process catalog. The CLI
still **selects** which names become model-visible (`surface=local` today).

## Layers (short)

```text
DEFINE+REGISTER   @tool → ToolDefinition + catalog entry (source=plugin:…)
SELECT            catalog.select(surface=…, …) → list[ToolDefinition]
EXECUTE           ToolRegistry (confirm, instance/session bind, invoke)
```

| Layer | Question |
|--------|----------|
| Register | What tools exist in this process? |
| Select | Which may the model see this session? |
| Execute | How are they confirmed and run? |

Builtins use the same path with `source=builtin`. Plugins must not reuse a
builtin (or another plugin) **tool name** — registration fails with a
collision error.

## Author a plugin package

### 1. Define tools with `@tool`

```python
# acme_plyngent/tools.py
from plyngent.agent import ToolTag, tool


@tool(name="acme_ping", tags=ToolTag.LOCAL)
async def acme_ping() -> str:
    """Return a fixed pong (example plugin tool)."""
    return "pong"
```

Notes:

- Prefer **`async def`** handlers (builtins are async-first).
- Default **`tags=ToolTag.LOCAL`**. Set at least one of `LOCAL` or `PUBLIC`.
- Workspace/exec tools should stay **`LOCAL`** unless reviewed for a shared host.
- Session-only helpers may add **`PUBLIC`** and **`SESSION_STATE`** when they
  only touch session data and host-approved APIs (see plans under `doc/plans/`).
- Optional soft-confirm bits: **`YOLO`** (eligible for YOLO auto-approve),
  **`TRUSTABLE`** (grant once per session after approve). Hard denylists are
  never YOLO-skipped.
- Docstring becomes the model-facing description; parameter schemas come from
  type hints.
- Use `register=False` only for unit tests that build a private registry.

### 2. Entry point

Call a zero-arg loader (or import a module) so `@tool` runs under the plugin
registration source:

```python
# acme_plyngent/__init__.py
def load() -> None:
    """Import tool modules so @tool registers into the process catalog."""
    from acme_plyngent import tools as _tools

    _ = _tools  # registration side effects
```

Declare the entry point (PEP 621 / setuptools / hatch style):

```toml
# pyproject.toml of the plugin package
[project]
name = "acme-plyngent"
version = "0.1.0"
dependencies = [
  "plyngent",  # or pin a compatible range
]

[project.entry-points."plyngent.tools"]
acme = "acme_plyngent:load"
```

- Entry-point **name** (`acme`) is the **plugin id** used in config allowlists
  and in catalog metadata (`plugin:acme`).
- Value is `module:attr`. If `attr` is callable, plyngent calls it after load;
  if it is a module, import alone is enough when tools registered at import time.

Install the plugin into the **same environment** as `plyngent` (editable fine):

```bash
pdm add acme-plyngent
# or: pip install -e ./path/to/acme-plyngent
```

## Enable plugins in config

In the user config (`plyngent config path`), under **`[agent]`**:

```toml
[agent]
# Allowlist of entry-point names. Default empty = load no plugins.
tool_plugins = ["acme"]
# tool_plugins = ["*"]          # every discovered plyngent.tools entry point
# Never load these names even if listed or matched by *:
# tool_plugins_disable = ["legacy"]
```

| Setting | Meaning |
|---------|---------|
| `tool_plugins = []` / omitted | Load **no** plugins (safe default). |
| `tool_plugins = ["acme", "other"]` | Load only those entry-point names. |
| `tool_plugins = ["*"]` | Load all discovered plugins. |
| `tool_plugins_disable = ["x"]` | Skip `x` even when allowlisted or `*`. |

Restart the chat REPL after changing config so the tool registry rebuilds.

CLI load order (simplified):

1. `register_builtin_tools()`
2. `load_plugin_tools(tool_plugins, disable=tool_plugins_disable)`
3. `catalog.select(surface="local")` → `ToolRegistry`

Inspect the model surface in the REPL:

```text
/tools --list
```

Listed tools show **tags** and **catalog source** (`builtin` vs `plugin:…`).

## Policy checklist for authors

1. Prefer **LOCAL** for host FS, shell, PTY, or arbitrary network side effects.
2. Do not shadow builtin names (`read_file`, `run_command`, todo tools, …).
3. Do not rely on process-global workspace/todo alone when a host binds
   instance/session context — prefer the same patterns as builtins
   (`get_workspace_root` already prefers instance policy; session state via
   context when you need it).
4. Soft danger: annotate with `YOLO` / `TRUSTABLE` only when the product
   confirm story matches; hard denylists stay in workspace/command policy.
5. Keep tools **async**; raise ordinary exceptions or return `error: …` strings
   consistently with builtins.

## Surfaces (local vs public)

| Catalog select | Included tools |
|----------------|----------------|
| `surface="local"` (CLI) | `LOCAL` and/or `PUBLIC` |
| `surface="public"` | `PUBLIC` only |

There is **no** multi-tenant public host in-tree yet; `surface=public` is
available for hosts that want a PUBLIC-only select (builtins: mainly the
todo series). Plugins should not set `PUBLIC` on workspace/exec tools by
default.

## API reference (code)

| Piece | Module |
|-------|--------|
| `@tool`, `ToolTag`, `ToolRegistry` | `plyngent.agent` / `plyngent.agent.tools` |
| Catalog, `ToolSource`, `select` | `plyngent.tools.catalog` |
| `load_plugin_tools` | `plyngent.tools.plugins` |
| Config fields | `AgentConfig.tool_plugins`, `tool_plugins_disable` |

## Related docs

- [architecture.md](./architecture.md) — package layout
- [plyngent.example.toml](./plyngent.example.toml) — config sample
- Design notes (may lag code): [plans/](./plans/) — registration, tags, PUBLIC surface
