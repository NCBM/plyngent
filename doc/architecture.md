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
    - router: Multi-source capability routing and interception.
        - interceptors: Capability of stealing a few arguments for partial re-routing.
        - ...
    - config: Plyngent configuration center.
    - agent: Agent capabilities and controlling.
    - tools: Client code of tool calls.
    - cli: CLI for chat/agent application.
    - web: Web service for chat/agent application.

