"""PersistentDataView transaction behavior."""

from __future__ import annotations

import pytest

from plyngent.agent.todo_stack import TodoStack
from plyngent.tools.view import MemoryViewStore, PersistentDataView


async def test_view_txn_commit() -> None:
    store = MemoryViewStore({})
    root: PersistentDataView[dict[str, object]] = PersistentDataView(store, bound_type=dict)
    async with root as data:
        data["todo"].store({"groups": [], "next_id": 1})
    loaded = await store.load()
    assert isinstance(loaded, dict)
    assert "todo" in loaded


async def test_view_rollback_on_error() -> None:
    store = MemoryViewStore({"keep": 1})
    root: PersistentDataView[dict[str, object]] = PersistentDataView(store)
    with pytest.raises(RuntimeError, match="boom"):
        async with root as data:
            data.store({"keep": 2})
            msg = "boom"
            raise RuntimeError(msg)
    assert await store.load() == {"keep": 1}


async def test_typed_todo_stack_commits_raw() -> None:
    store = MemoryViewStore({})
    root: PersistentDataView[dict[str, object]] = PersistentDataView(store, bound_type=dict)
    async with root as data:
        todo = data["todo"].typed(TodoStack)
        _ = todo.push_group(["A", "B"])
        assert todo.depth == 1
    loaded = await store.load()
    assert isinstance(loaded, dict)
    raw_todo = loaded.get("todo")
    assert isinstance(raw_todo, dict)
    assert "groups" in raw_todo
    restored = TodoStack.from_raw(raw_todo)
    assert restored.depth == 1
    assert [i.title for i in restored.groups[0].items] == ["A", "B"]


async def test_typed_todo_stack_roundtrip_from_raw() -> None:
    store = MemoryViewStore({"todo": {"groups": [], "next_id": 1}})
    root: PersistentDataView[dict[str, object]] = PersistentDataView(store, bound_type=dict)
    async with root as data:
        todo = data["todo"].typed(TodoStack)
        _ = todo.push_group(["only"])
    loaded = await store.load()
    assert isinstance(loaded, dict)
    again = TodoStack.from_raw(loaded["todo"])
    assert [i.title for i in again.all_items()] == ["only"]


async def test_mutate_outside_txn_errors() -> None:
    store = MemoryViewStore({})
    root: PersistentDataView[dict[str, object]] = PersistentDataView(store)
    with pytest.raises(RuntimeError, match="transaction"):
        root.store({"a": 1})
