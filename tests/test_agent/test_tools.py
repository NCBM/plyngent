from __future__ import annotations

from plyngent.agent import ToolRegistry, schema_from_callable, tool


def test_tool_decorator_infers_schema() -> None:
    @tool(register=False)
    def add(a: int, b: int = 0) -> int:
        """Add two numbers."""
        return a + b

    assert add.name == "add"
    assert add.description == "Add two numbers."
    assert add.parameters["type"] == "object"
    assert add.parameters["properties"]["a"] == {"type": "integer"}
    assert add.parameters["properties"]["b"] == {"type": "integer"}
    assert add.parameters["required"] == ["a"]


def test_tool_decorator_overrides() -> None:
    @tool(name="ping", description="health", register=False)
    def health() -> str:
        return "ok"

    assert health.name == "ping"
    assert health.description == "health"


def test_schema_from_callable_optional() -> None:
    def f(x: str | None = None) -> str:
        return x or ""

    schema = schema_from_callable(f)
    assert schema["properties"]["x"] == {"type": "string"}
    assert "required" not in schema


async def test_registry_execute_sync_and_async() -> None:
    @tool(register=False)
    def echo(text: str) -> str:
        return text

    @tool(register=False)
    async def upper(text: str) -> str:
        return text.upper()

    registry = ToolRegistry([echo, upper])
    assert await registry.execute("echo", '{"text": "hi"}') == "hi"
    assert await registry.execute("upper", '{"text": "hi"}') == "HI"


async def test_registry_unknown_and_bad_json() -> None:
    @tool(register=False)
    def noop() -> str:
        return "ok"

    registry = ToolRegistry([noop])
    assert "unknown" in await registry.execute("missing", "{}")
    assert "invalid" in await registry.execute("noop", "not-json")


async def test_registry_handler_error() -> None:
    @tool(register=False)
    def boom() -> str:
        msg = "nope"
        raise RuntimeError(msg)

    registry = ToolRegistry([boom])
    result = await registry.execute("boom", "{}")
    assert "failed" in result


def test_tool_items() -> None:
    @tool(register=False)
    def f(x: int) -> int:
        return x

    items = ToolRegistry([f]).tool_items()
    assert len(items) == 1
    assert items[0].function.name == "f"


def test_schema_list_and_bool() -> None:
    def g(flags: list[bool], *, ok: bool) -> None:
        del flags, ok

    schema = schema_from_callable(g)
    assert schema["properties"]["flags"]["type"] == "array"
    assert schema["properties"]["flags"]["items"] == {"type": "boolean"}
    assert schema["properties"]["ok"] == {"type": "boolean"}
