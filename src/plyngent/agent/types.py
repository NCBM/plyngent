"""Agent-level type aliases for LLM client dispatch."""

from __future__ import annotations

from plyngent.lmproto.anthropic.client import AnthropicClient
from plyngent.lmproto.deepseek.openai_compat.client import DeepseekOpenAIClient
from plyngent.lmproto.openai.client import OpenAIClient
from plyngent.lmproto.openai_compatible.client import OpenAICompatibleClient

type AnyLLMClient = OpenAICompatibleClient | DeepseekOpenAIClient | OpenAIClient | AnthropicClient
