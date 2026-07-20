from __future__ import annotations

from dataclasses import dataclass

# Product defaults when provider omits ``timeout`` (connect vs read idle).
DEFAULT_HTTP_CONNECT_TIMEOUT = 10.0
DEFAULT_HTTP_READ_TIMEOUT = 600.0

type HttpTimeout = float | tuple[float, float]


@dataclass
class OpenAIConfig:
    access_key_or_token: str
    base_url: str = "https://api.openai.com/v1"
    # Passed to ``niquests.AsyncSession(timeout=...)``.
    # ``float`` = single timeout; ``(connect, read)`` = split; factory always sets a concrete value.
    timeout: HttpTimeout = (DEFAULT_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT)
