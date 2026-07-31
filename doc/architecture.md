# Code Architecture

- plyngent
    - typedef: Shared type aliases.
    - lmproto: Protocols for interacting with LLM service.
        - (common)
            - model: Messages models of specified providers.
            - config: Configurations for specified providers.
            - client: Clients for accessing existent specified services.
            - server: Servers for accepting other clients.
        - openai_compatible: Chat Completions base (``kind=chat_completions``).
        - openai: Platform Responses + chat (``kind=responses``).
        - anthropic: Messages API (``kind=messages``).
        - deepseek
            - openai_compat (``kind=chat_completions``)
            - responses (``kind=responses``; DeepSeek Responses API)
            - anthropic (``kind=messages``; DeepSeek Anthropic-compatible API on
              ``https://api.deepseek.com/anthropic``)
    - utils: Common utilities for code architecture.
        - components: Utilities for class composition.
    - memory: Storage controlling for sessions and messages.
    - router: Multi-source capability routing (Phase H; not implemented).
    - config: Plyngent configuration center (TOML), including ``[plugins]`` and
      ``[networking]`` (e.g. fetch SSRF Fake-IP CIDR exemptions). Model entries
      may override provider ``preset`` / ``url`` for mixed API routing.
    - runtime: ``create_client`` maps effective provider preset → protocol client.
    - agent: Kind-based tool loop, streaming, usage, compact; bridges:
      ``responses_bridge``/``responses_dispatch``, ``messages_bridge``/``messages_dispatch``;
      ``@tool`` / tags / registry. History stays chat-completions-shaped.
    - tools: Workspace file/process/VCS/chat/todo/net tools; catalog; plugins;
      instance/session context and views.
    - prompting: Shared ask/choose/form for CLI and tools.
    - cli: Click entry, slash registry, REPL, one-shot chat.
    - web: Web service (Phase H; not implemented).

Plugins (third-party entry points, allowlisted under ``[plugins]``): [plugins.md](./plugins.md).

