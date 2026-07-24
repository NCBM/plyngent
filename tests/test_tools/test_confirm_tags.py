"""Tag-aware soft confirm: YOLO bit + TRUSTABLE grants."""

from __future__ import annotations

from plyngent.agent import ToolRegistry, ToolTag, tool
from plyngent.tools.context import SessionState, bind_tool_context
from plyngent.tools.grants import clear_grants, has_grant, hydrate_grants, persist_grants
from plyngent.tools.view import MemoryViewStore, session_data_view


async def test_yolo_only_skips_yolo_tagged_tools() -> None:
    calls: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.YOLO, register=False)
    async def yolo_ok() -> str:
        return "y"

    @tool(tags=ToolTag.LOCAL, register=False)
    async def no_yolo() -> str:
        return "n"

    def danger(name: str, _args: object) -> str | None:
        return "soft reason"

    async def confirm(name: str, _args: object, _reason: str) -> bool:
        calls.append(name)
        return True

    reg = ToolRegistry(
        [yolo_ok, no_yolo],
        danger=danger,
        on_confirm=confirm,
        yolo=True,
    )
    assert await reg.execute("yolo_ok", "{}") == "y"
    assert calls == []
    assert await reg.execute("no_yolo", "{}") == "n"
    assert calls == ["no_yolo"]


async def test_trustable_grant_once() -> None:
    calls: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.TRUSTABLE, register=False)
    async def trust_me() -> str:
        return "ok"

    def danger(name: str, _args: object) -> str | None:
        return "soft"

    async def confirm(name: str, _args: object, _reason: str) -> bool:
        calls.append(name)
        return True

    session = SessionState(session_id=1)
    reg = ToolRegistry(
        [trust_me],
        danger=danger,
        on_confirm=confirm,
        yolo=False,
        auto_bind_state=True,
        session_state=session,
    )
    with bind_tool_context(session=session):
        assert await reg.execute("trust_me", "{}") == "ok"
        assert await reg.execute("trust_me", "{}") == "ok"
    assert calls == ["trust_me"]


async def test_untagged_trustable_prompts_every_time() -> None:
    calls: list[str] = []

    @tool(tags=ToolTag.LOCAL, register=False)
    async def always_ask() -> str:
        return "ok"

    def danger(name: str, _args: object) -> str | None:
        return "soft"

    async def confirm(name: str, _args: object, _reason: str) -> bool:
        calls.append(name)
        return True

    reg = ToolRegistry([always_ask], danger=danger, on_confirm=confirm, yolo=False)
    assert await reg.execute("always_ask", "{}") == "ok"
    assert await reg.execute("always_ask", "{}") == "ok"
    assert calls == ["always_ask", "always_ask"]


async def test_trustable_grant_persists_to_session_data() -> None:
    calls: list[str] = []

    @tool(tags=ToolTag.LOCAL | ToolTag.TRUSTABLE, register=False)
    async def trust_me() -> str:
        return "ok"

    def danger(_name: str, _args: object) -> str | None:
        return "soft"

    async def confirm(name: str, _args: object, _reason: str) -> bool:
        calls.append(name)
        return True

    store = MemoryViewStore({})
    session = SessionState(session_id=1, data=session_data_view(store=store))
    reg = ToolRegistry(
        [trust_me],
        danger=danger,
        on_confirm=confirm,
        yolo=False,
        session_state=session,
    )
    assert await reg.execute("trust_me", "{}") == "ok"
    assert has_grant(session, "trust_me")
    loaded = await store.load()
    assert isinstance(loaded, dict)
    grants = loaded.get("grants")
    assert isinstance(grants, dict)
    assert grants.get("trust_me") is True

    # Fresh session + same store: hydrate restores the grant (no second prompt).
    session2 = SessionState(session_id=1, data=session_data_view(store=store))
    await hydrate_grants(session2)
    assert has_grant(session2, "trust_me")
    reg2 = ToolRegistry(
        [trust_me],
        danger=danger,
        on_confirm=confirm,
        yolo=False,
        session_state=session2,
    )
    assert await reg2.execute("trust_me", "{}") == "ok"
    assert calls == ["trust_me"]


async def test_clear_grants_and_persist() -> None:
    store = MemoryViewStore({})
    session = SessionState(session_id=2, data=session_data_view(store=store))
    session.add_grant("tool_a")
    await persist_grants(session)
    clear_grants(session)
    assert not has_grant(session, "tool_a")
    # Live clear does not rewrite the store; explicit persist does.
    await persist_grants(session)
    loaded = await store.load()
    assert isinstance(loaded, dict)
    assert loaded.get("grants") == {}
