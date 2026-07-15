from __future__ import annotations

from datetime import UTC, datetime

from plyngent.cli.export import (
    encode_session_export_json,
    format_session_export_md,
    session_export_payload,
)
from plyngent.lmproto.openai_compatible.model import AssistantChatMessage, UserChatMessage


def test_format_session_export_md() -> None:
    text = format_session_export_md(
        [
            UserChatMessage(content="hi"),
            AssistantChatMessage(content="yo", reasoning_content="plan"),
        ],
        sid=7,
        name="demo",
        workspace="/tmp/ws",
    )
    assert "# Session 7: demo" in text
    assert "## user" in text
    assert "hi" in text
    assert "### reasoning" in text
    assert "plan" in text
    assert "yo" in text


def test_session_export_json_roundtrip() -> None:
    messages = [UserChatMessage(content="a")]
    now = datetime.now(UTC)
    payload = session_export_payload(
        sid=7,
        name="demo",
        workspace="/tmp/ws",
        created_at=now,
        updated_at=now,
        messages=messages,
        provider_name="local",
        model="tiny",
    )
    assert payload.get("provider") == "local"
    assert payload.get("model") == "tiny"
    assert isinstance(payload, dict)
    raw = encode_session_export_json(payload)
    assert "7" in raw
    assert "demo" in raw
