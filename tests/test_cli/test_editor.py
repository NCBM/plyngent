from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from plyngent.cli.app import main
from plyngent.cli.editor import (
    ensure_config_file,
    get_editor,
    open_in_editor,
    prompt_edit_config,
    resolve_config_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_get_editor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    assert get_editor() is None
    monkeypatch.setenv("EDITOR", "  codium --wait  ")
    assert get_editor() == "codium --wait"


def test_resolve_config_path_override(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    assert resolve_config_path(path) == path


def test_ensure_config_file_creates_template(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "plyngent.toml"
    ensure_config_file(path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "database" in text


def test_open_in_editor_splits_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "plyngent.toml"
    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool = False) -> object:  # noqa: FBT001, FBT002
        del check
        calls.append(list(argv))

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    open_in_editor(path, editor="codium --wait")
    assert len(calls) == 1
    assert calls[0][0] == "codium"
    assert calls[0][1] == "--wait"
    assert calls[0][2] == str(path)
    assert path.is_file()


def test_open_in_editor_missing_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    with pytest.raises(Exception, match="EDITOR"):
        open_in_editor(tmp_path / "x.toml", editor=None)


def test_prompt_edit_no_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EDITOR", raising=False)
    assert prompt_edit_config(tmp_path / "p.toml") is False


def test_prompt_edit_declined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDITOR", "true")

    def _confirm(*_a: object, **_k: object) -> bool:
        return False

    monkeypatch.setattr("click.confirm", _confirm)
    assert prompt_edit_config(tmp_path / "p.toml", reason="empty") is False


def test_prompt_edit_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "p.toml"
    monkeypatch.setenv("EDITOR", "true")

    def _confirm(*_a: object, **_k: object) -> bool:
        return True

    monkeypatch.setattr("click.confirm", _confirm)
    opened: list[Path] = []

    def fake_open(p: Path, *, editor: str | None = None) -> None:
        del editor
        opened.append(p)

    monkeypatch.setattr("plyngent.cli.editor.open_in_editor", fake_open)
    assert prompt_edit_config(path, reason="No providers.") is True
    assert opened == [path]


def test_config_path_command(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["config", "path", "--config", str(tmp_path / "a.toml")])
    assert result.exit_code == 0
    assert str(tmp_path / "a.toml") in result.output


def test_config_edit_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "edit.toml"
    monkeypatch.setenv("EDITOR", "true")
    calls: list[list[str]] = []

    def fake_run(argv: list[str], check: bool = False) -> object:  # noqa: FBT001, FBT002
        del check
        calls.append(list(argv))

        class Result:
            returncode: int = 0

        return Result()

    monkeypatch.setattr("plyngent.cli.editor.subprocess.run", fake_run)
    runner = CliRunner()
    result = runner.invoke(main, ["config", "edit", "--config", str(path)])
    assert result.exit_code == 0
    assert calls and calls[0][-1] == str(path)
