from __future__ import annotations

import json

from plyngent.agent import ToolRegistry
from plyngent.prompting import NonInteractiveBackend, temporary_backend
from plyngent.tools.chat import CHAT_TOOLS, ask_user, choose_user, form_user
from tests.test_prompting import ScriptedBackend


async def test_ask_user_tool() -> None:
    backend = ScriptedBackend(["42"])
    with temporary_backend(backend):
        registry = ToolRegistry([ask_user])
        out = await registry.execute("ask_user", '{"question": "Answer?"}')
    assert out == "42"


async def test_choose_user_tool_index() -> None:
    backend = ScriptedBackend(["1"])
    with temporary_backend(backend):
        registry = ToolRegistry([choose_user])
        out = await registry.execute(
            "choose_user",
            json.dumps(
                {
                    "question": "Pick",
                    "options": json.dumps(["alpha", "beta"]),
                    "allow_custom": False,
                }
            ),
        )
    assert out == "alpha"


async def test_choose_user_bad_options() -> None:
    registry = ToolRegistry([choose_user])
    out = await registry.execute(
        "choose_user",
        json.dumps({"question": "Pick", "options": "not-json"}),
    )
    assert out.startswith("error:")


async def test_form_user_tool() -> None:
    backend = ScriptedBackend(["ncbm"], confirms=[True])
    fields = json.dumps([{"name": "user", "prompt": "User?"}])
    with temporary_backend(backend):
        registry = ToolRegistry([form_user])
        out = await registry.execute(
            "form_user",
            json.dumps({"title": "Setup", "fields": fields, "confirm_submit": True}),
        )
    assert json.loads(out) == {"user": "ncbm"}


async def test_chat_tools_in_default_list() -> None:
    names = {t.name for t in CHAT_TOOLS}
    assert names == {"ask_user", "choose_user", "form_user"}


async def test_ask_user_non_interactive_error() -> None:
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([ask_user])
        out = await registry.execute("ask_user", '{"question": "hi"}')
    assert out.startswith("error:")
