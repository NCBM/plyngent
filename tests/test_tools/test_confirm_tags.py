"""Tag-aware soft confirm: YOLO bit + TRUSTABLE grants."""

from __future__ import annotations

from plyngent.agent import ToolRegistry, ToolTag, tool
from plyngent.tools.context import SessionState, bind_tool_context


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
