"""A tool's own shape check answers a call before anyone is asked about it."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from plyngent.agent import ToolRegistry
from plyngent.tools.file.edit_lineno import edit_lineno
from plyngent.tools.file.edit_replace import edit_replace
from plyngent.tools.net.fetch_local import fetch_local
from plyngent.tools.process.open_pty import open_pty
from plyngent.tools.process.write_pty_keys import write_pty_keys

if TYPE_CHECKING:
    from collections.abc import Mapping

    from plyngent.agent import ToolDefinition


def _everything_needs_a_confirm(name: str, _args: Mapping[str, object]) -> str:
    """A danger classifier that prompts for every call, so a suppressed prompt is visible."""
    return f"run {name}"


def _registry(definition: ToolDefinition) -> tuple[ToolRegistry, list[str]]:
    asked: list[str] = []

    def on_confirm(name: str, _args: Mapping[str, object], _reason: str) -> bool:
        asked.append(name)
        return True

    return ToolRegistry([definition], danger=_everything_needs_a_confirm, on_confirm=on_confirm), asked


@pytest.mark.parametrize(
    ("definition", "args", "expected"),
    [
        # A string where the tool needs an argv array would splat into characters.
        (open_pty, {"command": "ls -la"}, "only accepts an array of args"),
        # A typo in a <key> marker must not land in a shell prompt.
        (write_pty_keys, {"session_id": 1, "data": "<nope>"}, "unknown key <nope>"),
        (edit_replace, {"path": "a.py", "old_string": "", "new_string": "x"}, "old_string must not be empty"),
        (
            edit_replace,
            {"path": "a.py", "old_string": "x", "new_string": "y", "max_replaces": 0},
            "max_replaces must be >= 1",
        ),
        (
            edit_lineno,
            {"path": "a.py", "start_line": 0, "end_line": 1, "new_content": "x"},
            "start_line must be >= 1",
        ),
        (
            edit_lineno,
            {"path": "a.py", "start_line": 3, "end_line": 2, "new_content": "x"},
            "end_line must be >= start_line",
        ),
        (fetch_local, {"url": "https://example.com", "method": "PATCH"}, "method must be one of"),
        (fetch_local, {"url": "ftp://example.com"}, "url scheme must be http or https"),
        (fetch_local, {"url": "https://example.com", "timeout_seconds": "soon"}, "timeout_seconds must be a number"),
    ],
)
async def test_a_bad_argument_is_answered_before_the_prompt(
    definition: ToolDefinition,
    args: dict[str, object],
    expected: str,
) -> None:
    registry, asked = _registry(definition)
    out = await registry.execute(definition.name, json.dumps(args))
    assert asked == []
    assert expected in out


async def test_a_call_that_can_run_still_asks() -> None:
    """The check answers only what it can judge; a runnable call keeps its prompt."""
    registry, asked = _registry(write_pty_keys)
    out = await registry.execute("write_pty_keys", json.dumps({"session_id": 1, "data": "<enter>"}))
    assert asked == ["write_pty_keys"]
    assert out.startswith(("[approval]", "error:"))
