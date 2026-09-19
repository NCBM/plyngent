from __future__ import annotations

import importlib
import json

from plyngent.agent import ToolRegistry
from plyngent.prompting import NonInteractiveBackend, temporary_backend
from plyngent.tools.chat import CHAT_TOOLS, ask_user, choose_user, form_user, wait
from tests.test_prompting import ScriptedBackend

wait_module = importlib.import_module("plyngent.tools.chat.wait")


async def test_ask_user_tool() -> None:
    backend = ScriptedBackend(["42"])
    with temporary_backend(backend):
        registry = ToolRegistry([ask_user])
        out = await registry.execute("ask_user_line", '{"question": "Answer?"}')
    assert out == "42"


async def test_choose_user_tool_index() -> None:
    backend = ScriptedBackend(["1"])
    with temporary_backend(backend):
        registry = ToolRegistry([choose_user])
        out = await registry.execute(
            "ask_user_choice",
            json.dumps(
                {
                    "question": "Pick",
                    "options": ["alpha", "beta"],
                    "allow_custom": False,
                }
            ),
        )
    assert out == "alpha"


async def test_choose_user_dict_options() -> None:
    backend = ScriptedBackend(["1"])
    with temporary_backend(backend):
        registry = ToolRegistry([choose_user])
        out = await registry.execute(
            "ask_user_choice",
            json.dumps(
                {
                    "question": "Pick",
                    "options": [{"label": "yes", "description": "affirm"}],
                }
            ),
        )
    assert out == "yes"


async def test_choose_user_invalid_options() -> None:
    registry = ToolRegistry([choose_user])
    out = await registry.execute(
        "ask_user_choice",
        json.dumps({"question": "Pick", "options": [42]}),
    )
    assert "error" in out


async def test_form_user_tool() -> None:
    backend = ScriptedBackend(["ncbm"], confirms=[True])
    with temporary_backend(backend):
        registry = ToolRegistry([form_user])
        out = await registry.execute(
            "ask_user_form",
            json.dumps(
                {
                    "title": "Setup",
                    "fields": [{"name": "user", "prompt": "User?"}],
                    "confirm_submit": True,
                }
            ),
        )
    assert json.loads(out) == {"user": "ncbm"}


async def test_chat_tools_in_default_list() -> None:
    names = {t.name for t in CHAT_TOOLS}
    assert names == {"ask_user_line", "ask_user_choice", "ask_user_form", "wait"}


async def test_ask_user_non_interactive_error() -> None:
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([ask_user])
        out = await registry.execute("ask_user_line", '{"question": "hi"}')
    assert out.startswith("error:")


async def test_wait_tool_zero_seconds(monkeypatch) -> None:
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 0}')
    assert out == "waited 0s"


async def test_wait_tool_negative_duration() -> None:
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": -1}')
    assert out.startswith("error:")


async def test_wait_tool_noninteractive_sleeps(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(wait_module.asyncio, "sleep", fake_sleep)
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 5}')
    assert out == "waited 5s"
    assert slept == [5]


async def test_wait_tool_disturbed_with_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        wait_module,
        "read_line_with_timeout",
        lambda prompt, timeout: "tests are slow",
    )
    with temporary_backend(ScriptedBackend([])):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 3}')
    assert out == "disturbed by user: tests are slow"


async def test_wait_tool_disturbed_empty_enter(monkeypatch) -> None:
    monkeypatch.setattr(wait_module, "read_line_with_timeout", lambda prompt, timeout: "")
    with temporary_backend(ScriptedBackend([])):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 3}')
    assert out == "disturbed by user (no reason)"


async def test_wait_tool_times_out(monkeypatch) -> None:
    monkeypatch.setattr(wait_module, "read_line_with_timeout", lambda prompt, timeout: None)
    with temporary_backend(ScriptedBackend([])):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 3}')
    assert out == "waited 3s"


async def test_wait_tool_prompt_two_lines_with_reason(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(
        wait_module,
        "read_line_with_timeout",
        lambda prompt, timeout: prompts.append(prompt) or None,
    )
    with temporary_backend(ScriptedBackend([])):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 5, "reason": "tests are slow"}')
    assert out == "waited 5s"
    assert prompts == ["Waiting for 5s (tests are slow). Press Enter to disturb.\nReason (optional): "]


async def test_wait_tool_prompt_two_lines_no_reason(monkeypatch) -> None:
    prompts: list[str] = []
    monkeypatch.setattr(
        wait_module,
        "read_line_with_timeout",
        lambda prompt, timeout: prompts.append(prompt) or None,
    )
    with temporary_backend(ScriptedBackend([])):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", '{"duration": 5}')
    assert out == "waited 5s"
    assert prompts == ["Waiting for 5s. Press Enter to disturb.\nReason (optional): "]


def test_wait_prompt_plain_when_not_a_tty() -> None:
    """Prompt styling is stripped when stdout is not a TTY (pytest capture)."""
    prompt = wait_module._wait_prompt(5, reason="tests")
    assert "\x1b[" not in prompt
    assert prompt == "Waiting for 5s (tests). Press Enter to disturb.\nReason (optional): "


def test_wait_prompt_colored_on_tty(monkeypatch) -> None:
    """On a TTY the prompt is styled: cyan status, yellow input prompt."""

    class _FakeStdout:
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(wait_module.sys, "stdout", _FakeStdout())
    prompt = wait_module._wait_prompt(5, reason="tests")
    assert "\x1b[36m" in prompt  # cyan status
    assert "\x1b[33m" in prompt  # yellow input prompt


def test_shape_of_names_json_types() -> None:
    from plyngent.tools.chat.shape import first_error, shape_of

    assert shape_of("x") == "a string"
    assert shape_of([]) == "an array"
    assert shape_of({}) == "an object"
    assert shape_of(True) == "a boolean"  # bool is not reported as a number
    assert shape_of(3) == "a number"
    assert shape_of(3.5) == "a number"
    assert shape_of(None) == "null"
    assert first_error(None, "boom", "later") == "boom"
    assert first_error(None, None) is None


async def test_ask_user_line_rejects_non_string_args() -> None:
    """Mis-typed args get explained instead of reaching the human as garbage."""
    backend = ScriptedBackend(["unused"])
    with temporary_backend(backend):
        registry = ToolRegistry([ask_user])
        bad_question = await registry.execute("ask_user_line", json.dumps({"question": ["a", "b"]}))
        bad_default = await registry.execute(
            "ask_user_line",
            json.dumps({"question": "Which?", "default": 5}),
        )
    assert bad_question.startswith("error:")
    assert "`question`" in bad_question
    assert "got an array" in bad_question
    assert bad_default.startswith("error:")
    assert "`default`" in bad_default
    assert "got a number" in bad_default
    assert backend.lines == ["unused"]  # the prompt never happened


async def test_choose_user_rejects_string_options() -> None:
    """``options="alpha, beta"`` used to build a menu of single characters."""
    backend = ScriptedBackend(["1"])
    with temporary_backend(backend):
        registry = ToolRegistry([choose_user])
        out = await registry.execute(
            "ask_user_choice",
            json.dumps({"question": "Pick", "options": "alpha, beta"}),
        )
    assert out.startswith("error:")
    assert "`options`" in out
    assert "not a string" in out
    assert '["alpha", "beta"]' in out
    assert backend.lines == ["1"]


async def test_choose_user_rejects_object_options() -> None:
    registry = ToolRegistry([choose_user])
    out = await registry.execute(
        "ask_user_choice",
        json.dumps({"question": "Pick", "options": {"alpha": 1}}),
    )
    assert out.startswith("error:")
    assert "got an object" in out


async def test_form_user_rejects_string_fields() -> None:
    backend = ScriptedBackend(["ncbm"], confirms=[True])
    with temporary_backend(backend):
        registry = ToolRegistry([form_user])
        out = await registry.execute(
            "ask_user_form",
            json.dumps({"title": "Setup", "fields": '[{"name": "user"}]'}),
        )
    assert out.startswith("error:")
    assert "`fields`" in out
    assert "not a string" in out
    assert backend.lines == ["ncbm"]


async def test_form_user_rejects_non_object_field() -> None:
    registry = ToolRegistry([form_user])
    out = await registry.execute(
        "ask_user_form",
        json.dumps({"title": "Setup", "fields": ["user"]}),
    )
    assert out == "error: each field must be an object with name/prompt"


async def test_wait_accepts_numeric_string_duration(monkeypatch) -> None:
    """A model sending "5" instead of 5 still waits."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(wait_module.asyncio, "sleep", fake_sleep)
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", json.dumps({"duration": "5"}))
    assert out == "waited 5s"
    assert slept == [5.0]


async def test_wait_accepts_fractional_duration(monkeypatch) -> None:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(wait_module.asyncio, "sleep", fake_sleep)
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", json.dumps({"duration": 2.5}))
    assert out == "waited 2.5s"
    assert slept == [2.5]


async def test_wait_rejects_bad_duration_shapes() -> None:
    cases: list[tuple[object, str]] = [
        ("soon", "non-numeric string"),
        ([5], "an array"),
        ({"seconds": 5}, "an object"),
        (True, "a boolean"),
    ]
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        for value, expected in cases:
            out = await registry.execute("wait", json.dumps({"duration": value}))
            assert out.startswith("error:"), f"{value!r} was accepted: {out!r}"
            assert "`duration`" in out
            assert expected in out, f"{value!r}: {out!r}"


async def test_wait_rejects_non_string_reason() -> None:
    with temporary_backend(NonInteractiveBackend()):
        registry = ToolRegistry([wait])
        out = await registry.execute("wait", json.dumps({"duration": 0, "reason": ["why"]}))
    assert out.startswith("error:")
    assert "`reason`" in out


def test_wait_prompt_renders_fractional_seconds() -> None:
    prompt = wait_module._wait_prompt(2.5, reason=None)
    assert prompt.startswith("Waiting for 2.5s.")
    assert wait_module._wait_prompt(5, reason=None).startswith("Waiting for 5s.")


async def test_choose_user_reports_missing_options() -> None:
    """A call without the required ``options`` lists what the tool accepts."""
    backend = ScriptedBackend(["1"])
    with temporary_backend(backend):
        registry = ToolRegistry([choose_user])
        out = await registry.execute("ask_user_choice", json.dumps({"question": "Pick"}))
    assert out.startswith("error:")
    assert "missing required argument(s): options" in out
    assert "it accepts: allow_custom, default, options, question" in out
    assert backend.lines == ["1"]  # the human was never asked


async def test_form_user_reports_missing_fields() -> None:
    registry = ToolRegistry([form_user])
    out = await registry.execute("ask_user_form", json.dumps({"title": "Setup"}))
    assert out.startswith("error:")
    assert "missing required argument(s): fields" in out


async def test_wait_reports_missing_duration() -> None:
    registry = ToolRegistry([wait])
    out = await registry.execute("wait", "{}")
    assert out.startswith("error:")
    assert "missing required argument(s): duration" in out
