from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, cast, overload

import msgspec

from ...openai_compatible.client import BaseOpenAIClient, read_response_body
from ...openai_compatible.compat import coerce_chat_completions_param_any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ...openai_compatible.config import OpenAIConfig
    from ...openai_compatible.model import ChatCompletionChunk, ChatCompletionResponse
    from .model import ChatCompletionsParam


def _inject_thinking(data: bytes, param: ChatCompletionsParam) -> bytes:
    """Inject the ``thinking`` flag into the encoded request body.

    DeepSeek's reasoning model requires the ``thinking`` parameter to be
    explicitly set (default is ``enabled``); ``reasoning_effort = "none"`` is the
    way to ask for it off. The agent loop constructs the base
    ``ChatCompletionsParam`` which lacks this field, so we add it after msgspec
    encoding.
    """
    body = json.loads(data)
    body["thinking"] = {"type": "disabled" if param.reasoning_effort == "none" else "enabled"}
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def _inject_missing_reasoning_content(data: bytes) -> bytes:
    """Give every assistant message a ``reasoning_content`` field.

    Thinking mode rejects an assistant round whose message carries no reasoning
    ("the ``reasoning_content`` … must be passed back to the API") — for a tool
    call and for a plain answer alike. History can hold such a round: a
    rolled-back turn keeps the calls that ran as a bare pairing carrier without
    it (see ``rollback_tail``), a synthetic ``todo_list`` nag is forged the same
    way, and a compacted session starts with a seed summary carrying none. The
    field only has to be there, so an empty string satisfies the check (unlike
    the Responses API, whose ``reasoning`` item needs non-empty text — see
    ``agent.responses_bridge``).
    """
    body = cast("dict[str, object]", json.loads(data))
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in cast("list[object]", messages):
            if not isinstance(message, dict):
                continue
            item = cast("dict[str, object]", message)
            if item.get("role") != "assistant" or "reasoning_content" in item:
                continue
            item["reasoning_content"] = ""
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


class DeepseekOpenAIClient(BaseOpenAIClient):
    kind: str = "chat_completions"

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)

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
        param = coerce_chat_completions_param_any(msgspec.structs.replace(param, stream=stream))
        data = _inject_thinking(self.encoder.encode(param), param)
        data = _inject_missing_reasoning_content(data)
        if stream:
            resp = await self.session.post(
                "/chat/completions",
                data=data,
                headers={"Content-Type": "application/json"},
                stream=True,
            )
            return self._parse_sse(resp)
        resp = await self.session.post(
            "/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            stream=False,
        )
        await self._ensure_ok(resp)
        body = await read_response_body(resp)
        if body is None:
            msg = "chat completions response body is empty"
            raise RuntimeError(msg)
        if not isinstance(body, (bytes, bytearray)):
            msg = f"chat completions response body has unexpected type {type(body)!r}"
            raise TypeError(msg)
        return self.decoder.decode(bytes(body))
