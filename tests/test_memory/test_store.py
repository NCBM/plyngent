from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import AssistantChatMessage, UserChatMessage
from plyngent.memory import DEFAULT_USER_NAME, MemoryStore

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@pytest.fixture
async def store() -> AsyncIterator[MemoryStore]:
    memory = await MemoryStore.open(DatabaseConfig())
    yield memory
    await memory.close()


async def test_open_creates_default_user(store: MemoryStore) -> None:
    user = await store.get_user_by_name(DEFAULT_USER_NAME)
    assert user is not None
    assert user.name == DEFAULT_USER_NAME


async def test_create_session_uses_default_user(store: MemoryStore) -> None:
    session = await store.create_session(name="chat-1")
    assert session.sid is not None
    assert session.name == "chat-1"
    user = await store.get_user_by_name(DEFAULT_USER_NAME)
    assert user is not None
    assert session.uid == user.uid


async def test_append_and_list_messages(store: MemoryStore) -> None:
    session = await store.create_session()
    user_msg = UserChatMessage(content="hello")
    assistant_msg = AssistantChatMessage(content="hi")

    row0 = await store.append_message(session.sid, user_msg)
    row1 = await store.append_message(session.sid, assistant_msg)
    assert row0.seq == 0
    assert row1.seq == 1

    messages = await store.list_messages(session.sid)
    assert len(messages) == 2  # noqa: PLR2004
    assert isinstance(messages[0], UserChatMessage)
    assert messages[0].content == "hello"
    assert isinstance(messages[1], AssistantChatMessage)
    assert messages[1].content == "hi"


async def test_list_sessions(store: MemoryStore) -> None:
    s1 = await store.create_session(name="a")
    s2 = await store.create_session(name="b")
    sessions = await store.list_sessions()
    ids = {s.sid for s in sessions}
    assert s1.sid in ids
    assert s2.sid in ids
