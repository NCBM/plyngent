"""Todo tools via session.data PersistentDataView (session-bound only)."""

from __future__ import annotations

from plyngent.agent import ToolRegistry
from plyngent.agent.todo_stack import TodoStack
from plyngent.tools.context import SessionState, bind_tool_context
from plyngent.tools.todo import TODO_TOOLS
from plyngent.tools.view import MemoryViewStore, session_data_view


async def test_todo_tools_session_data() -> None:
    store = MemoryViewStore({})
    session = SessionState(session_id="s1", data=session_data_view(store=store))
    registry = ToolRegistry(list(TODO_TOOLS), session_state=session)

    out = await registry.execute("todo_push", '{"titles": ["A", "B"]}')
    assert "pushed" in out
    assert session.todo is not None
    assert session.todo.depth == 1

    loaded = await store.load()
    assert isinstance(loaded, dict)
    raw = loaded.get("todo")
    assert isinstance(raw, dict)
    assert "groups" in raw
    restored = TodoStack.from_raw(raw)
    assert [i.title for i in restored.groups[0].items] == ["A", "B"]

    out2 = await registry.execute("todo_list", "{}")
    assert "A" in out2 and "B" in out2


async def test_todo_tools_prefer_session_todo_facet() -> None:
    stack = TodoStack()
    store = MemoryViewStore({})
    session = SessionState(session_id="s2", data=session_data_view(store=store), todo=stack)
    registry = ToolRegistry(list(TODO_TOOLS), session_state=session)

    _ = await registry.execute("todo_push", '{"titles": ["X"]}')
    assert stack.depth == 1
    assert session.todo is stack

    loaded = await store.load()
    assert isinstance(loaded, dict)
    assert isinstance(loaded.get("todo"), dict)


async def test_todo_tools_view_isolation_two_sessions() -> None:
    store_a = MemoryViewStore({})
    store_b = MemoryViewStore({})
    session_a = SessionState(session_id="a", data=session_data_view(store=store_a))
    session_b = SessionState(session_id="b", data=session_data_view(store=store_b))

    reg_a = ToolRegistry(list(TODO_TOOLS), session_state=session_a)
    reg_b = ToolRegistry(list(TODO_TOOLS), session_state=session_b)

    _ = await reg_a.execute("todo_push", '{"titles": ["only-a"]}')
    _ = await reg_b.execute("todo_push", '{"titles": ["only-b"]}')

    loaded_a = await store_a.load()
    loaded_b = await store_b.load()
    assert isinstance(loaded_a, dict)
    assert isinstance(loaded_b, dict)
    raw_a = TodoStack.from_raw(loaded_a.get("todo"))
    raw_b = TodoStack.from_raw(loaded_b.get("todo"))
    assert [i.title for i in raw_a.all_items()] == ["only-a"]
    assert [i.title for i in raw_b.all_items()] == ["only-b"]
    assert session_a.todo is not session_b.todo


async def test_with_bound_context_without_registry_session() -> None:
    """Handlers honor contextvars when registry does not hold session_state."""
    store = MemoryViewStore({})
    session = SessionState(session_id="ctx", data=session_data_view(store=store))
    registry = ToolRegistry(list(TODO_TOOLS), auto_bind_state=False)
    with bind_tool_context(session=session):
        out = await registry.execute("todo_push", '{"titles": ["via-ctx"]}')
    assert "pushed" in out
    loaded = await store.load()
    assert isinstance(loaded, dict)
    assert isinstance(loaded.get("todo"), dict)


async def test_session_on_todo_change_fires() -> None:
    hits: list[str] = []

    def session_hook() -> None:
        hits.append("session")

    stack = TodoStack()
    store = MemoryViewStore({})
    session = SessionState(
        session_id="hook",
        data=session_data_view(store=store),
        todo=stack,
        on_todo_change=session_hook,
    )
    registry = ToolRegistry(list(TODO_TOOLS), session_state=session)
    _ = await registry.execute("todo_push", '{"titles": ["H"]}')
    assert hits == ["session"]


async def test_todo_without_session_errors() -> None:
    registry = ToolRegistry(list(TODO_TOOLS))
    out = await registry.execute("todo_list", "{}")
    assert "error" in out.lower()
