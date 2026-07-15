# Code Architecture

- plyngent
    - typedef: Shared type aliases.
    - lmproto: Protocols for interacting with LLM service.
        - (common)
            - model: Messages models of specified providers.
            - config: Configurations for specified providers.
            - client: Clients for accessing existent specified services.
            - server: Servers for accepting other clients.
        - openai
        - openai_compatible
        - anthropic
        - ollama
        - deepseek
            - openai_compat
            - anthropic
    - utils: Common utilities for code architecture.
        - components: Utilities for class composition.
    - memory: Storage controlling for sessions and messages.
    - router: Multi-source capability routing (Phase H; not implemented).
    - config: Plyngent configuration center (TOML).
    - agent: Tool loop, streaming, usage, compact.
    - tools: Workspace file/process/VCS/chat tools.
    - prompting: Shared ask/choose/form for CLI and tools.
    - cli: Click entry, slash registry, REPL, one-shot chat.
    - web: Web service (Phase H; not implemented).

