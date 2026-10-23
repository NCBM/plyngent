from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import pytest
from click.testing import CliRunner

from plyngent.cli.app import _read_prompt_text, main
from plyngent.cli.exit_codes import EXIT_OK
from plyngent.lmproto.openai_compatible.model import (
    AssistantChatMessage,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatCompletionsParam,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def test_read_prompt_flag_only() -> None:
    assert _read_prompt_text("hello", stdin_isatty=True) == "hello"
    assert _read_prompt_text("  ", stdin_isatty=True) is None
    assert _read_prompt_text(None, stdin_isatty=True) is None


def test_read_prompt_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStdin:
        def read(self) -> str:
            return "from stdin\n"

    monkeypatch.setattr("plyngent.cli.app.sys.stdin", FakeStdin())
    assert _read_prompt_text(None, stdin_isatty=False) == "from stdin"
    assert _read_prompt_text("flag", stdin_isatty=False) == "flag\nfrom stdin"


def test_chat_oneshot_requires_provider_flags(tmp_path: Path) -> None:
    config = tmp_path / "plyngent.toml"
    _ = config.write_text(
        """
[providers.a]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk"

[providers.a.models]
"m1" = {}

[providers.b]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk"

[providers.b.models]
"m2" = {}
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["chat", "--config", str(config), "-p", "hi", "--workspace", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "provider" in result.output.lower() or "provider" in (result.stderr or "").lower()


def test_chat_oneshot_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "plyngent.toml"
    # Ephemeral file DB under tmp_path — never the user's durable chat.db.
    # (Unset/empty [database].url is rewritten to ~/.local/share/plyngent/chat.db.)
    db_path = tmp_path / "test-chat.db"
    _ = config.write_text(
        f"""
[database]
implementation = "sqlite"
url = "{db_path.as_posix()}"

[providers.local]
preset = "openai-compatible"
url = "https://example.com/v1"
access_key_or_token = "sk"

[providers.local.models]
"tiny" = {{}}
""",
        encoding="utf-8",
    )

    class DummyClient:
        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[False] = False
        ) -> ChatCompletionResponse: ...

        @overload
        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: Literal[True]
        ) -> AsyncIterator[ChatCompletionChunk]: ...

        async def chat_completions(
            self, param: ChatCompletionsParam, *, stream: bool = False
        ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
            del param
            if stream:

                async def empty() -> AsyncIterator[ChatCompletionChunk]:
                    if False:
                        yield
                    return

                return empty()
            return ChatCompletionResponse(
                id="1",
                object="chat.completion",
                created=0,
                model="tiny",
                choices=[
                    ChatCompletionChoice(
                        index=0,
                        message=AssistantChatMessage(content="pong"),
                        logprobs={},
                        finish_reason="stop",
                    )
                ],
                system_fingerprint="",
                usage={},
            )

    monkeypatch.setattr("plyngent.cli.app.create_client", lambda _p, **_kw: DummyClient())
    monkeypatch.setattr(
        "plyngent.cli.state.create_client",
        lambda _p, **_kw: DummyClient(),
    )

    # The durable default must stay untouched, and the test must not read the
    # host's own chat.db to prove it: point that default into tmp_path and check
    # nothing was created there.
    unused_data_dir = tmp_path / "user-data"
    _ = unused_data_dir.mkdir()
    monkeypatch.setattr("plyngent.cli.app.user_data_path", lambda *_args, **_kwargs: unused_data_dir)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "chat",
            "--config",
            str(config),
            "--provider",
            "local",
            "--model",
            "tiny",
            "-p",
            "ping",
            "--no-stream",
            "--workspace",
            str(tmp_path),
            "--quiet",
        ],
    )
    assert result.exit_code == EXIT_OK
    assert "pong" in result.output
    assert db_path.is_file()
    assert not (unused_data_dir / "chat.db").exists()


def test_chat_help_mentions_prompt() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--prompt" in result.output or "-p" in result.output
    assert "Exit codes" in result.output or "one-shot" in result.output.lower()
