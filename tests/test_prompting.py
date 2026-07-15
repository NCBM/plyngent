from __future__ import annotations

import pytest

from plyngent.prompting import (
    ChoiceOption,
    FormField,
    NonInteractiveBackend,
    NonInteractiveError,
    ask,
    choose,
    confirm,
    form,
    temporary_backend,
)


class ScriptedBackend:
    """Deterministic backend for unit tests."""

    def __init__(self, lines: list[str], *, confirms: list[bool] | None = None) -> None:
        self.lines = list(lines)
        self.confirms = list(confirms or [])
        self.interactive = True
        self.echoes: list[str] = []

    def is_interactive(self) -> bool:
        return self.interactive

    def read_line(self, prompt: str, *, default: str | None = None) -> str:
        del prompt
        if self.lines:
            return self.lines.pop(0)
        if default is not None:
            return default
        msg = "no scripted lines left"
        raise NonInteractiveError(msg)

    def confirm(self, prompt: str, *, default: bool = False) -> bool:
        del prompt
        if self.confirms:
            return self.confirms.pop(0)
        return default

    def echo(self, message: str = "", *, err: bool = False) -> None:
        del err
        self.echoes.append(message)

    def secho(self, message: str, *, fg: str | None = None, err: bool = False) -> None:
        del fg, err
        self.echoes.append(message)


def test_ask_free_text() -> None:
    backend = ScriptedBackend(["hello world"])
    with temporary_backend(backend):
        assert ask("Name?") == "hello world"


def test_ask_default_when_empty_uses_backend_default() -> None:
    backend = ScriptedBackend([])
    with temporary_backend(backend):
        assert ask("Name?", default="anon") == "anon"


def test_choose_by_index() -> None:
    backend = ScriptedBackend(["2"])
    with temporary_backend(backend):
        assert (
            choose(
                "Pick",
                [
                    ChoiceOption(label="a", value="A"),
                    ChoiceOption(label="b", value="B"),
                ],
                allow_custom=False,
            )
            == "B"
        )


def test_choose_by_label() -> None:
    backend = ScriptedBackend(["keep"])
    with temporary_backend(backend):
        assert choose("Pick", ["keep", "abort"], allow_custom=False) == "keep"


def test_choose_custom_text() -> None:
    backend = ScriptedBackend(["something else"])
    with temporary_backend(backend):
        assert choose("Pick", ["a", "b"], allow_custom=True) == "something else"


def test_choose_rejects_custom_when_disabled() -> None:
    backend = ScriptedBackend(["nope", "1"])
    with temporary_backend(backend):
        assert choose("Pick", ["a", "b"], allow_custom=False) == "a"


def test_confirm_yes() -> None:
    backend = ScriptedBackend([], confirms=[True])
    with temporary_backend(backend):
        assert confirm("ok?", default=False) is True


def test_form_with_confirm() -> None:
    backend = ScriptedBackend(["alice", "2"], confirms=[True])
    with temporary_backend(backend):
        answers = form(
            "Profile",
            [
                FormField(name="name", prompt="Name?"),
                FormField(
                    name="role",
                    prompt="Role?",
                    options=["dev", "ops"],
                    allow_custom=False,
                ),
            ],
            confirm_submit=True,
        )
    assert answers == {"name": "alice", "role": "ops"}


def test_non_interactive_ask_requires_default() -> None:
    with temporary_backend(NonInteractiveBackend()), pytest.raises(NonInteractiveError):
        _ = ask("Name?")


def test_non_interactive_ask_uses_default() -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert ask("Name?", default="x") == "x"


def test_non_interactive_confirm_uses_default() -> None:
    with temporary_backend(NonInteractiveBackend()):
        assert confirm("ok?", default=False) is False
        assert confirm("ok?", default=True) is True
