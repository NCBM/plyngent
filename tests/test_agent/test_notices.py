from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

from plyngent.agent import ChatAgent, Notice, aside_notice, format_moment, resume_notice
from plyngent.agent.chat import incomplete_turn_user_text
from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    AssistantFunctionTool,
    AssistantFunctionToolCall,
    DeveloperChatMessage,
    SystemChatMessage,
    ToolChatMessage,
    UserChatMessage,
)
from plyngent.memory import MemoryStore
from tests.test_agent.test_loop import ScriptedClient


def test_notice_message_carries_marker_and_kind() -> None:
    message = Notice(kind="resume", body="  hello there  ").to_message()
    assert isinstance(message, DeveloperChatMessage)
    assert message.content == "[plyngent notice: resume]\nhello there"


def test_format_moment_localizes_and_labels_missing() -> None:
    assert format_moment(None) == "unknown"
    aware = datetime(2026, 9, 14, 22, 9, tzinfo=UTC)
    assert format_moment(aware) == aware.astimezone().strftime("%Y-%m-%d %H:%M")
    # SQLite hands back naive datetimes for tz-aware columns; read them as UTC.
    assert format_moment(aware.replace(tzinfo=None)) == format_moment(aware)
    offset = datetime(2026, 9, 14, 22, 9, tzinfo=timezone(timedelta(hours=8)))
    assert format_moment(offset) == offset.astimezone().strftime("%Y-%m-%d %H:%M")


def test_resume_notice_states_what_is_gone_and_restored() -> None:
    notice = resume_notice(last_active=datetime(2026, 9, 14, 22, 9, tzinfo=UTC))
    assert notice.kind == "resume"
    assert "resumed in a new plyngent process" in notice.body
    assert "PTY sessions" in notice.body
    assert "grants were restored" in notice.body
    assert notice.summary.startswith("resumed in a new process")
    assert format_moment(datetime(2026, 9, 14, 22, 9, tzinfo=UTC)) in notice.summary


def test_aside_notice_matches_tool_scope() -> None:
    read = aside_notice(tools_mode="read")
    assert read.kind == "aside"
    assert "read-only" in read.body
    assert "not saved" in read.body
    assert "read tools" in read.summary

    assert "Tools are disabled" in aside_notice(tools_mode="no").body
    assert "Full tools are available" in aside_notice(tools_mode="full").body
    # Unknown mode degrades to the read-only wording rather than an empty notice.
    assert aside_notice(tools_mode="weird").body == read.body


async def test_push_notice_appends_and_persists() -> None:
    store = await MemoryStore.open(DatabaseConfig())
    try:
        session = await store.create_session(name="t")
        _ = await store.append_message(session.sid, SystemChatMessage(content="sys"))
        _ = await store.append_message(session.sid, UserChatMessage(content="hi"))
        agent = ChatAgent(ScriptedClient([]), model="m", memory=store, session_id=session.sid)
        await agent.load_history()

        message = await agent.push_notice(resume_notice(last_active=None))

        assert agent.messages[-1] is message
        assert agent.persist_from == len(agent.messages)  # never re-persisted by a batch
        loaded = await store.list_messages(session.sid)
        assert isinstance(loaded[-1], DeveloperChatMessage)
        assert loaded[-1].content.startswith("[plyngent notice: resume]")
    finally:
        await store.close()


def test_trailing_developer_message_keeps_incomplete_turn_detectable() -> None:
    """A pushed notice must not hide a turn that can still be retried."""
    notice = resume_notice(last_active=None).to_message()
    assert incomplete_turn_user_text([UserChatMessage(content="do it"), notice]) == "do it"
    batch = [
        UserChatMessage(content="do it"),
        AssistantChatMessage(
            content="",
            tool_calls=[
                AssistantFunctionToolCall(
                    id="1",
                    function=AssistantFunctionTool(name="ping", arguments="{}"),
                )
            ],
        ),
        ToolChatMessage(content="pong", tool_call_id="1"),
        notice,
    ]
    assert incomplete_turn_user_text(batch) == "do it"
    # A finished turn (trailing assistant) stays non-retryable past a notice.
    done = [UserChatMessage(content="do it"), AssistantChatMessage(content="ok"), notice]
    assert incomplete_turn_user_text(done) is None
