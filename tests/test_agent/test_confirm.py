from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.agent import ToolRegistry, tool
from plyngent.tools.danger import classify_danger

if TYPE_CHECKING:
    from collections.abc import Mapping


@tool
def delete_path(path: str, *, recursive: bool = False) -> str:
    del recursive
    return f"deleted {path}"


@tool
def read_file(path: str) -> str:
    return f"read {path}"


async def test_confirm_allow() -> None:
    decisions: list[str] = []

    def on_confirm(name: str, args: Mapping[str, object], reason: str) -> bool:
        del args
        decisions.append(f"{name}:{reason}")
        return True

    registry = ToolRegistry(
        [delete_path, read_file],
        danger=classify_danger,
        on_confirm=on_confirm,
    )
    assert await registry.execute("delete_path", '{"path": "a.txt"}') == "deleted a.txt"
    assert len(decisions) == 1
    assert await registry.execute("read_file", '{"path": "a.txt"}') == "read a.txt"
    assert len(decisions) == 1


async def test_confirm_deny() -> None:
    def on_confirm(name: str, args: Mapping[str, object], reason: str) -> bool:
        del name, args, reason
        return False

    registry = ToolRegistry(
        [delete_path],
        danger=classify_danger,
        on_confirm=on_confirm,
    )
    out = await registry.execute("delete_path", '{"path": "a.txt"}')
    assert "denied" in out
    assert "delete" in out


async def test_no_hooks_skips_confirm() -> None:
    registry = ToolRegistry([delete_path])
    assert registry.soft_confirm is False
    assert await registry.execute("delete_path", '{"path": "a.txt"}') == "deleted a.txt"


async def test_soft_confirm_property() -> None:
    def on_confirm(name: str, args: Mapping[str, object], reason: str) -> bool:
        del name, args, reason
        return True

    gated = ToolRegistry([delete_path], danger=classify_danger, on_confirm=on_confirm)
    assert gated.soft_confirm is True
