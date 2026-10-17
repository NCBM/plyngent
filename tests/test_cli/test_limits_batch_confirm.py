"""Aggregated soft-confirm prompt: one box for a round's tool calls."""

from __future__ import annotations

from plyngent.agent import BatchConfirmDecision, ToolConfirmRequest
from plyngent.cli.limits import (
    _prompt_confirm_batch_sync,
    format_batch_confirm_box,
    format_tool_confirm_box,
    prompt_confirm_batch_async,
)
from plyngent.prompting import NonInteractiveBackend, temporary_backend
from tests.test_prompting import ScriptedBackend


def _requests() -> list[ToolConfirmRequest]:
    return [
        ToolConfirmRequest(name="delete_path", args={"path": "a.txt"}, reason="delete path 'a.txt'"),
        ToolConfirmRequest(name="move_path", args={"src": "b", "dst": "c"}, reason="move 'b' → 'c'"),
    ]


def test_batch_box_numbers_every_call() -> None:
    box = format_batch_confirm_box(_requests())
    lines = box.splitlines()
    assert "confirm · 2 tool calls" in lines[1]
    assert any("[delete_path] delete path 'a.txt'" in line for line in lines)
    assert any("[move_path] move 'b' → 'c'" in line for line in lines)
    # Numbering keeps the listed order visible when the user answers.
    assert any(line.startswith("│ 1. ") for line in lines)
    assert any(line.startswith("│ 2. ") for line in lines)


def test_batch_box_indents_continuation_lines() -> None:
    box = format_batch_confirm_box(
        [
            ToolConfirmRequest(name="run_argv", args={"argv": ["python", "-c", "x"]}, reason="argv\n  body"),
            ToolConfirmRequest(name="delete_path", args={"path": "a"}, reason="delete path 'a'"),
        ]
    )
    assert any(line.startswith("│    ") and "body" in line for line in box.splitlines())


def test_single_call_box_is_unchanged() -> None:
    box = format_tool_confirm_box("delete_path", "delete path 'a.txt'")
    assert "confirm · tool 'delete_path'" in box.splitlines()[1]


def test_batch_prompt_allows_every_call_on_yes() -> None:
    with temporary_backend(ScriptedBackend(["y"])):
        decision = _prompt_confirm_batch_sync(_requests())
    assert decision == BatchConfirmDecision(allowed=True)


def test_batch_prompt_hands_the_round_to_the_per_call_prompt_on_manual() -> None:
    with temporary_backend(ScriptedBackend(["m"])):
        decision = _prompt_confirm_batch_sync(_requests())
    assert decision.each is True
    assert decision.allowed is False


def test_batch_prompt_denies_with_a_comment() -> None:
    with temporary_backend(ScriptedBackend(["", "keep those files"])):
        decision = _prompt_confirm_batch_sync(_requests())
    assert decision.allowed is False
    assert decision.comment == "keep those files"


def test_batch_prompt_defaults_to_deny() -> None:
    with temporary_backend(ScriptedBackend([])):
        decision = _prompt_confirm_batch_sync(_requests())
    assert decision == BatchConfirmDecision(allowed=False)


def test_batch_prompt_denies_when_not_interactive() -> None:
    with temporary_backend(NonInteractiveBackend()):
        decision = _prompt_confirm_batch_sync(_requests())
    assert decision == BatchConfirmDecision(allowed=False)


async def test_batch_prompt_async_allows() -> None:
    with temporary_backend(ScriptedBackend(["yes"])):
        decision = await prompt_confirm_batch_async(_requests())
    assert decision == BatchConfirmDecision(allowed=True)
