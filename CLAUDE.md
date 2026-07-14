# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Plyngent is an LLM chat and agent toolkit (Python 3.14+, PDM-managed). It is in early development — core protocol clients exist, while agent, CLI, web, memory, and config modules are planned but not yet implemented.

## Commands

```bash
pdm install          # first-time dependency setup
pdm sync             # sync after pulling changes

pdm run basedpyright .   # type checking (basedpyright, "recommended" strictness)
pdm run ruff check .     # linting
pdm run ruff format .    # formatting
```

There is no test runner configured yet.

## Architecture

### Data modeling: `msgspec.Struct`

All protocol models use `msgspec.Struct` — not dataclasses, not Pydantic. This gives type-safe, schema-validated structs with efficient JSON codec support. Use `msgspec.field(default=UNSET)` for optional fields to distinguish "not provided" from `None`.

The `Unset` type alias (`typedef.py`) wraps `msgspec.UNSET` as `Literal[UnsetType.UNSET]`. `JSONSchema` is `dict[str, Any]`.

### Protocol layering (`lmproto/`)

```
lmproto/openai_compatible/     ← Base: model, config, client
lmproto/deepseek/openai_compat/ ← Extends base via inheritance + extra fields
```

- **`openai_compatible/model.py`** — chat message models (`SystemChatMessage`, `UserChatMessage`, `AssistantChatMessage`, `ToolChatMessage`), tool definitions, request/response structs, streaming chunks. All `msgspec.Struct` with `rename="snake"` for camelCase JSON interop.
- **`openai_compatible/client.py`** — `BaseOpenAIClient` using `niquests` for async HTTP with SSE streaming. Two response modes: `ChatCompletionResponse` (non-streaming) and `AsyncIterator[ChatCompletionChunk]` (streaming).
- **`openai_compatible/config.py`** — `OpenAIConfig` dataclass (base URL, API key, model name, max tokens, temperature).
- **DeepSeek extension** — `DeepseekOpenAIClient` extends `BaseOpenAIClient`. Models add `reasoning_content`, `prefix`, `ThinkingOptions`, and `DeepSeekReasoningEffort` (including `"max"`).

### Composition utility: `Forward` descriptor

`utils/components.py` provides a `Forward[T]` descriptor for forwarding attribute access to a composed sub-object — prefer composition over inheritance where appropriate. The `forward()` factory function constructs `Forward` instances with type inference.

### Type annotations are mandatory

Basedpyright in `recommended` mode (strictest preset). Ruff lint rules include `ANN` (flake8-annotations) — all functions must have type annotations except private functions (`ANN202` suppressed). Use PEP 695 `type X = ...` syntax for type aliases. Python 3.14+ generics syntax is expected.

## Commit messages

Scoped commit messages, not conventional commits:

```
<scope>: <brief description>
```

Examples: `deps: add fastapi`, `core/mq: fix incorrect message sending`, `router: add routing service for xxx`, `test/webserver: change test client`, `ci/lint: run ruff style check`.

Check `git log` for the full convention.
