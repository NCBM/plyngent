"""Directive checkpoint bands and append-only inject."""

from __future__ import annotations

from plyngent.agent.directive_checkpoint import (
    bands_to_fire,
    inject_directive_checkpoints,
    parse_checkpoint_bands,
    token_band,
)
from plyngent.lmproto.openai_compatible.model import (
    AnyChatMessage,
    DeveloperChatMessage,
    UserChatMessage,
)


def test_token_band() -> None:
    assert token_band(0, 100_000) == 0
    assert token_band(99_999, 100_000) == 0
    assert token_band(100_000, 100_000) == 1
    assert token_band(250_000, 100_000) == 2
    assert token_band(100_000, 0) == 0


def test_bands_to_fire_fill_gaps() -> None:
    assert bands_to_fire(last_fired_band=0, current_band=0) == []
    assert bands_to_fire(last_fired_band=0, current_band=1) == [1]
    assert bands_to_fire(last_fired_band=0, current_band=2) == [1, 2]
    assert bands_to_fire(last_fired_band=2, current_band=2) == []
    assert bands_to_fire(last_fired_band=1, current_band=3) == [2, 3]


def test_inject_append_only_and_parse() -> None:
    messages: list[AnyChatMessage] = [UserChatMessage(content="hi")]
    band, appended = inject_directive_checkpoints(
        messages,
        prompt_tokens=100_000,
        source="api",
        interval=100_000,
        last_fired_band=0,
    )
    assert band == 1
    assert len(appended) == 1
    assert isinstance(messages[-1], DeveloperChatMessage)
    assert "band=1" in messages[-1].content
    assert parse_checkpoint_bands(messages) == 1

    band2, appended2 = inject_directive_checkpoints(
        messages,
        prompt_tokens=250_000,
        source="api",
        interval=100_000,
        last_fired_band=band,
    )
    assert band2 == 2
    assert len(appended2) == 1
    assert parse_checkpoint_bands(messages) == 2

    # Same band: no re-fire
    band3, appended3 = inject_directive_checkpoints(
        messages,
        prompt_tokens=250_000,
        source="api",
        interval=100_000,
        last_fired_band=band2,
    )
    assert band3 == 2
    assert appended3 == []


def test_inject_disabled() -> None:
    messages: list[AnyChatMessage] = []
    band, appended = inject_directive_checkpoints(
        messages,
        prompt_tokens=500_000,
        source="api",
        interval=0,
        last_fired_band=0,
    )
    assert band == 0
    assert appended == []
