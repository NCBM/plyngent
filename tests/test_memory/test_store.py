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


async def test_session_workspace_binding(store: MemoryStore, tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    a = tmp_path / "proj-a"
    b = tmp_path / "proj-b"
    a.mkdir()
    b.mkdir()
    sa = await store.create_session(name="a", workspace=a)
    sb = await store.create_session(name="b", workspace=b)
    assert sa.workspace == str(a.resolve())
    listed_a = await store.list_sessions(workspace=a)
    assert {s.sid for s in listed_a} == {sa.sid}
    listed_b = await store.list_sessions(workspace=b)
    assert {s.sid for s in listed_b} == {sb.sid}


async def test_workspace_column_migration(tmp_path: object) -> None:
    """Existing DBs without session.workspace get the column via ALTER."""
    from pathlib import Path

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    assert isinstance(tmp_path, Path)
    db_path = tmp_path / "legacy.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        _ = await conn.execute(
            text(
                "CREATE TABLE user ("
                "uid INTEGER PRIMARY KEY, name VARCHAR(48) UNIQUE, "
                "email VARCHAR(255) UNIQUE, password_hash VARCHAR(256), "
                "created_at DATETIME)"
            )
        )
        _ = await conn.execute(
            text(
                "CREATE TABLE session ("
                "sid INTEGER PRIMARY KEY, uid INTEGER, name VARCHAR(64), "
                "created_at DATETIME, updated_at DATETIME)"
            )
        )
        _ = await conn.execute(
            text(
                "CREATE TABLE message ("
                "mid INTEGER PRIMARY KEY, sid INTEGER, seq INTEGER, "
                "data JSON, created_at DATETIME, updated_at DATETIME)"
            )
        )
    await engine.dispose()

    store = await MemoryStore.open(DatabaseConfig(url=str(db_path)))
    session = await store.create_session(name="migrated", workspace=tmp_path)
    from plyngent.memory.database.store import normalize_workspace

    assert session.workspace == normalize_workspace(tmp_path)
    await store.close()
