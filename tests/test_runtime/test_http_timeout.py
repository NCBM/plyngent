from __future__ import annotations

import math

import pytest

from plyngent.config.models import (
    HttpTimeoutConfig,
    OpenAICompatibleProvider,
    OpenAIProvider,
)
from plyngent.lmproto.openai_compatible.config import (
    DEFAULT_HTTP_CONNECT_TIMEOUT,
    DEFAULT_HTTP_READ_TIMEOUT,
)
from plyngent.runtime import (
    InvalidHttpTimeoutError,
    ProviderNotSupportedError,
    create_client,
    normalize_http_timeout,
    provider_to_openai_config,
)


def test_normalize_none_uses_product_defaults() -> None:
    assert normalize_http_timeout(None) == (
        DEFAULT_HTTP_CONNECT_TIMEOUT,
        DEFAULT_HTTP_READ_TIMEOUT,
    )


def test_normalize_float() -> None:
    assert normalize_http_timeout(120) == 120.0
    assert normalize_http_timeout(0.5) == 0.5


def test_normalize_table_partial_and_full() -> None:
    assert normalize_http_timeout(HttpTimeoutConfig()) == (
        DEFAULT_HTTP_CONNECT_TIMEOUT,
        DEFAULT_HTTP_READ_TIMEOUT,
    )
    assert normalize_http_timeout(HttpTimeoutConfig(connect=5.0)) == (
        5.0,
        DEFAULT_HTTP_READ_TIMEOUT,
    )
    assert normalize_http_timeout(HttpTimeoutConfig(read=30.0)) == (
        DEFAULT_HTTP_CONNECT_TIMEOUT,
        30.0,
    )
    assert normalize_http_timeout(HttpTimeoutConfig(connect=3.0, read=90.0)) == (3.0, 90.0)


@pytest.mark.parametrize(
    "bad",
    [
        0,
        -1,
        math.nan,
        math.inf,
        True,  # bool is int subclass but rejected
        HttpTimeoutConfig(connect=0),
        HttpTimeoutConfig(read=-5),
        HttpTimeoutConfig(connect=math.nan),
    ],
)
def test_normalize_rejects_invalid(bad: float | HttpTimeoutConfig) -> None:
    with pytest.raises(InvalidHttpTimeoutError):
        _ = normalize_http_timeout(bad)


def test_provider_to_openai_config_default_timeout() -> None:
    provider = OpenAIProvider(access_key_or_token="sk-test")
    config = provider_to_openai_config(provider)
    assert config.timeout == (DEFAULT_HTTP_CONNECT_TIMEOUT, DEFAULT_HTTP_READ_TIMEOUT)


def test_provider_to_openai_config_float_timeout() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        timeout=45.0,
    )
    config = provider_to_openai_config(provider)
    assert config.timeout == 45.0


def test_provider_to_openai_config_table_timeout() -> None:
    provider = OpenAIProvider(
        access_key_or_token="sk-test",
        timeout=HttpTimeoutConfig(connect=2.0, read=99.0),
    )
    config = provider_to_openai_config(provider)
    assert config.timeout == (2.0, 99.0)


def test_invalid_timeout_via_create_client() -> None:
    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        timeout=0,
    )
    with pytest.raises(ProviderNotSupportedError, match="timeout"):
        _ = create_client(provider)


def test_client_session_receives_timeout() -> None:
    from plyngent.lmproto.openai_compatible import OpenAICompatibleClient

    provider = OpenAICompatibleProvider(
        access_key_or_token="sk-test",
        url="https://example.com/v1",
        timeout=HttpTimeoutConfig(connect=7.0, read=11.0),
    )
    client = create_client(provider)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.session.timeout == (7.0, 11.0)
