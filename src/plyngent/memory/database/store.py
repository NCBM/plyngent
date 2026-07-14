from __future__ import annotations

from typing import TYPE_CHECKING, Self

import msgspec
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from plyngent.config.models import DatabaseConfig
from plyngent.lmproto.openai_compatible.model import AnyChatMessage

from .engine import create_engine
from .schema import Message, PlyngentBase, Session, User

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


DEFAULT_USER_NAME = "local"
DEFAULT_USER_EMAIL = "local@localhost"
DEFAULT_USER_PASSWORD_HASH = ""


class MemoryStore:
    """Async persistence for users, chat sessions, and messages."""

    _engine: AsyncEngine
    _session_factory: async_sessionmaker[AsyncSession]

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._session_factory = async_sessionmaker(engine, expire_on_commit=False)

    @classmethod
    async def open(
        cls,
        database: DatabaseConfig | Mapping[str, object],
        *,
        echo: bool = False,
        init_schema: bool = True,
        ensure_default_user: bool = True,
    ) -> Self:
        """Open a store from database config, optionally creating tables and default user."""
        config = database if isinstance(database, DatabaseConfig) else msgspec.convert(dict(database), DatabaseConfig)
        store = cls(create_engine(config, echo=echo))
        if init_schema:
            await store.create_schema()
        if ensure_default_user:
            _ = await store.ensure_default_user()
        return store

    async def create_schema(self) -> None:
        """Create all tables if they do not exist."""
        async with self._engine.begin() as conn:
            _ = await conn.run_sync(PlyngentBase.metadata.create_all)

    async def close(self) -> None:
        """Dispose the underlying engine."""
        await self._engine.dispose()

    async def ensure_default_user(self) -> User:
        """Return the default local user, creating it if missing."""
        async with self._session_factory() as session:
            result = await session.execute(select(User).where(User.name == DEFAULT_USER_NAME))
            user = result.scalar_one_or_none()
            if user is not None:
                return user
            user = User(
                name=DEFAULT_USER_NAME,
                email=DEFAULT_USER_EMAIL,
                password_hash=DEFAULT_USER_PASSWORD_HASH,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user_by_name(self, name: str) -> User | None:
        async with self._session_factory() as session:
            result = await session.execute(select(User).where(User.name == name))
            return result.scalar_one_or_none()

    async def create_session(self, *, uid: int | None = None, name: str = "default") -> Session:
        """Create a chat session for ``uid`` (default local user when omitted)."""
        if uid is None:
            user = await self.ensure_default_user()
            uid = user.uid
        async with self._session_factory() as session:
            row = Session(uid=uid, name=name)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def get_session(self, sid: int) -> Session | None:
        async with self._session_factory() as session:
            return await session.get(Session, sid)

    async def list_sessions(self, *, uid: int | None = None) -> Sequence[Session]:
        async with self._session_factory() as session:
            stmt = select(Session).order_by(Session.sid)
            if uid is not None:
                stmt = stmt.where(Session.uid == uid)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def append_message(self, sid: int, message: AnyChatMessage) -> Message:
        """Append a chat message to a session with the next sequence number."""
        data = msgspec.to_builtins(message)
        if not isinstance(data, dict):
            msg = "chat message must serialize to a JSON object"
            raise TypeError(msg)

        async with self._session_factory() as session:
            result = await session.execute(
                select(Message.seq).where(Message.sid == sid).order_by(Message.seq.desc()).limit(1)
            )
            last_seq = result.scalar_one_or_none()
            seq = 0 if last_seq is None else last_seq + 1
            row = Message(sid=sid, seq=seq, data=data)
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row

    async def list_messages(self, sid: int) -> list[AnyChatMessage]:
        """Load chat messages for a session in sequence order."""
        async with self._session_factory() as session:
            result = await session.execute(select(Message).where(Message.sid == sid).order_by(Message.seq))
            rows = result.scalars().all()
        return [msgspec.convert(row.data, type=AnyChatMessage) for row in rows]

    async def list_message_rows(self, sid: int) -> Sequence[Message]:
        """Load raw message rows for a session in sequence order."""
        async with self._session_factory() as session:
            result = await session.execute(select(Message).where(Message.sid == sid).order_by(Message.seq))
            return result.scalars().all()
