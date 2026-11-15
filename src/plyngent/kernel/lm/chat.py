"""The chat-completions dialect: one flat list of roles, and the tool calls inside an assistant turn.

The oldest of the three shapes and the one every compatible gateway speaks, which is why it is the
dialect a local vLLM, a LiteLLM proxy or a vendor's OpenAI-compatible endpoint is reached over. It is
also the least precise: a conversation is a list of dictionaries keyed by ``role``, a tool call is a
member of the assistant message that asked for it, and a tool's answer is *another* message rather than
a run of blocks — so "which result answers which call" is carried by an id inside the tool message.

Two things this dialect cannot do, and they are said here rather than discovered later:

* **A picture has no channel.** Its only image slot is inside a *person's* message, and a tool's answer
  is not a person's; putting pixels there would be inventing a speaker. ``pictures`` is therefore
  ``False`` and a route on this dialect is blind however its caps read (see :class:`Codec`).
* **A thinking budget has no portable spelling.** ``responses`` carries an effort level and
  ``messages`` carries a token budget; here a vendor invents ``reasoning`` or ``thinking`` and the next
  one spells it differently. So the flag is not written at all — a route that wants thinking out loud
  belongs on one of the other two.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm.caps import PrefixCache
from plyngent.kernel.lm.codec import Codec, as_dict, as_int, as_list, as_text, decoded, schema
from plyngent.kernel.model import Assistant, Call, ModelFailed, Reply, Result, Usage, User

if TYPE_CHECKING:
    from typing import Any

    from plyngent.kernel.lm.caps import Caps
    from plyngent.kernel.lm.codec import Body
    from plyngent.kernel.model import Request

#: The word a route names this dialect by.
CHAT = "chat-completions"

#: What a ``finish_reason`` means when it is not "it ended on its own".
#:
#: ``tool_calls`` is deliberately absent: asking for a tool is not being cut off, and an answer that
#: asked for one is complete. ``content_filter`` is ``DENIED`` because the provider decided not to
#: answer, which no retry lifts.
_CUT: dict[str, Failure] = {"length": Failure.LIMIT, "content_filter": Failure.DENIED}


class Chat(Codec):
    """``POST {base}/chat/completions``: messages, tools, and one choice per answer."""

    name: str = CHAT
    endpoint: str = "/chat/completions"

    @override
    def encode(self, request: Request, *, model: str, caps: Caps) -> Body:
        body: Body = {
            "model": model,
            "messages": [*_system(request), *(_message(item) for item in request.messages)],
            "stream": False,
            "max_tokens": caps.max_output,
            "parallel_tool_calls": caps.parallel_tools,
        }
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": tool.name, "parameters": schema(tool.schema)}}
                for tool in request.tools
            ]
        if caps.prefix_cache is PrefixCache.IMPLICIT and request.cache_key:
            body["prompt_cache_key"] = request.cache_key
        return body

    @override
    def decode(self, payload: bytes) -> Reply:
        document = decoded(payload, where=self.name)
        choices = as_list(as_dict(document).get("choices"))
        if not choices:
            msg = f"{self.name} answered with no choices, so nothing was said"
            raise ModelFailed(Failure.PROTOCOL, msg)
        first = as_dict(choices[0])
        message = first.get("message")
        return Reply(
            text=_said(message),
            calls=_calls(message),
            usage=_usage(document),
            cut=_CUT.get(as_text(first.get("finish_reason"))),
        )

    @override
    def explain(self, payload: bytes) -> str:
        try:
            document = decoded(payload, where=self.name)
        except Exception:  # noqa: BLE001 — an error body that is not JSON is not a worse error
            return payload.decode("utf-8", "replace").strip()
        said = as_dict(document).get("error")
        if isinstance(said, str):
            return said
        return as_text(as_dict(said).get("message")) or as_text(as_dict(document).get("message"))


def _system(request: Request) -> list[Body]:
    """The part of the prompt that is not the conversation, as the first message of the list.

    A message here and not a field of its own, because this dialect has no field for one: the parts of
    a prompt that do not move are the front of the list, which is also why they sit before the tools in
    a cached prefix (§10.5).
    """
    return [{"role": "system", "content": request.system}] if request.system else []


def _message(message: User | Assistant | Result) -> Body:
    """One canonical message, in the spelling this dialect has for it."""
    if isinstance(message, User):
        return {"role": "user", "content": message.text}
    if isinstance(message, Assistant):
        said: Body = {"role": "assistant"}
        if message.text:
            said["content"] = message.text
        if message.calls:
            said["tool_calls"] = [
                {"id": call.id, "type": "function", "function": {"name": call.tool, "arguments": _spelled(call)}}
                for call in message.calls
            ]
        return said
    return {"role": "tool", "tool_call_id": message.call, "content": message.text}


def _spelled(call: Call) -> str:
    """A call's arguments as this dialect takes them: text, and never nothing at all."""
    return call.arguments or "{}"


def _said(message: Any) -> str:
    """What an answer said, which this dialect may spell as one string or as a list of parts."""
    content = as_dict(message).get("content")
    if isinstance(content, str):
        return content
    return "".join(as_text(as_dict(part).get("text")) for part in as_list(content))


def _calls(message: Any) -> tuple[Call, ...]:
    """The calls one answer asked for, in the order it asked for them."""
    calls: list[Call] = []
    for one in as_list(as_dict(message).get("tool_calls")):
        function = as_dict(as_dict(one).get("function"))
        if not function:
            continue
        calls.append(
            Call(
                id=as_text(as_dict(one).get("id")),
                tool=as_text(function.get("name")),
                arguments=as_text(function.get("arguments")),
            )
        )
    return tuple(calls)


def _usage(document: Any) -> Usage:
    """What the answer cost, with the cached share read from either spelling of it.

    Two vendors, two names for "this much of the prompt was already on their side": OpenAI nests it in
    ``prompt_tokens_details.cached_tokens`` and DeepSeek reports ``prompt_cache_hit_tokens``. The
    numbers are the point of §10.5, so both spellings are read here rather than left to a report that
    would silently say zero.
    """
    usage = as_dict(as_dict(document).get("usage"))
    details = as_dict(usage.get("prompt_tokens_details"))
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    return Usage(
        prompt=as_int(usage.get("prompt_tokens")),
        completion=as_int(usage.get("completion_tokens")),
        cached=None if cached is None else as_int(cached),
    )
