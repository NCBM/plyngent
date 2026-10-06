"""Anthropic API configuration."""

from __future__ import annotations

from dataclasses import dataclass

from plyngent.lmproto.openai_compatible.config import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_READ_TIMEOUT,
    HttpTimeout,
)

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class AnthropicConfig:
    api_key: str
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    timeout: HttpTimeout = (DEFAULT_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT)
    # Explicit ``thinking.budget_tokens`` for reasoning models (0 = derive it
    # from the configured reasoning_effort; see config/reasoning.py).
    thinking_budget_tokens: int = 0
