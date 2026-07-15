from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from plyngent.cli.app import main


def test_providers_list() -> None:
    config = Path(__file__).resolve().parents[1] / "test_config" / "plyngent-valid.toml"
    runner = CliRunner()
    result = runner.invoke(main, ["providers", "--config", str(config)])
    assert result.exit_code == 0
    assert "test1" in result.output
    assert "openai" in result.output


def test_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chat" in result.output
    assert "--log-level" in result.output


def test_invalid_config_toml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    _ = bad.write_text("this is = not [ valid toml\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(main, ["providers", "--config", str(bad)])
    assert result.exit_code != 0
    assert "invalid config" in result.output.lower() or "toml" in result.output.lower()
