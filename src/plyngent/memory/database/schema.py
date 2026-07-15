from datetime import datetime  # noqa: TC003

from sqlalchemy import JSON, DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class PlyngentBase(DeclarativeBase):
    pass


class User(PlyngentBase):
    __tablename__: str = "user"

    uid: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(48), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Session(PlyngentBase):
    __tablename__: str = "session"

    sid: Mapped[int] = mapped_column(primary_key=True)
    uid: Mapped[int] = mapped_column(ForeignKey(User.uid, ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    # Absolute workspace path this chat is bound to (tools root); null = legacy unbound.
    workspace: Mapped[str | None] = mapped_column(String(1024), nullable=True, index=True)
    # Last selected provider/model for this session (config provider key + model id).
    provider_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class Message(PlyngentBase):
    __tablename__: str = "message"
    __table_args__: tuple[object, ...] = (UniqueConstraint("sid", "seq", name="uq_session_seq"),)

    mid: Mapped[int] = mapped_column(primary_key=True)
    sid: Mapped[int] = mapped_column(ForeignKey(Session.sid, ondelete="CASCADE"), index=True)
    seq: Mapped[int] = mapped_column(index=True)
    data: Mapped[dict[str, object]] = mapped_column(JSON())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    session: Mapped[Session] = relationship(back_populates="messages")
