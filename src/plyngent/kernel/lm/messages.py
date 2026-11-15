"""The messages dialect: a system prompt of its own, and a conversation of content blocks.

Where ``chat`` flattens a conversation into roles, this one keeps it as blocks: an assistant turn is a
list that may hold text and the calls it asked for *in one message*, and a tool's answer is a
``tool_result`` block inside a **person's** message rather than a message of its own. Two consequences
shape this file:

* **A run of results is one message.** Consecutive tool answers have to arrive together, because this
  dialect wants one turn of each speaker at a time; the merge is here rather than in the caller, since
  "what counts as one turn" is the dialect's opinion.
* **A picture has a channel, and it is inside a tool's answer.** So ``pictures`` is ``True``: a result
  whose pixels were fetched carries them here as an ``image`` block beside the text, base64-encoded,
  which is what lets a route on this dialect honestly claim ``Caps.vision``.

Two more things are this dialect's alone and are read off ``Caps``: ``max_tokens`` is **required** —
there is no request without it — and a thinking budget is a token count rather than an effort level.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, override

import msgspec

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm.caps import PrefixCache
from plyngent.kernel.lm.codec import Codec, as_dict, as_int, as_list, as_text, decoded, schema, spelled
from plyngent.kernel.model import Assistant, Call, ModelFailed, Reply, Result, Usage, User
from plyngent.kernel.toolresult import Status

if TYPE_CHECKING:
    from typing import Any

    from plyngent.kernel.lm.caps import Caps
    from plyngent.kernel.lm.codec import Body
    from plyngent.kernel.model import AnyMessage, Request

#: The word a route names this dialect by.
MESSAGES = "messages"

#: What a ``stop_reason`` means when the answer did not simply end.
#:
#: ``tool_use`` and ``stop_sequence`` are absent because neither is being cut off: one asked for a
#: tool and the other met a stop string the caller supplied. ``refusal`` is ``DENIED`` — the provider
#: decided not to answer, and asking again would not change its mind.
_CUT: dict[str, Failure] = {"max_tokens": Failure.LIMIT, "refusal": Failure.DENIED}


class Messages(Codec):
    """``POST {base}/messages``: a system prompt, content blocks, and ``max_tokens`` always."""

    name: str = MESSAGES
    endpoint: str = "/messages"
    pictures: bool = True

    @override
    def encode(self, request: Request, *, model: str, caps: Caps) -> Body:
        body: Body = {
            "model": model,
            "messages": _messages(request.messages),
            "max_tokens": caps.max_output,
            # Not one this dialect can read yet (``sse`` is False), so it never asks for one.
            "stream": False,
        }
        if request.system:
            body["system"] = _system(request.system, caps)
        if request.tools:
            body["tools"] = _tools(request, caps)
        if caps.reasoning and caps.reasoning_budget:
            body["thinking"] = {"type": "enabled", "budget_tokens": caps.reasoning_budget}
        return body

    @override
    def decode(self, payload: bytes) -> Reply:
        document = decoded(payload, where=self.name)
        blocks = as_dict(document).get("content")
        if not isinstance(blocks, list):
            msg = f"{self.name} answered with content that is not a list of blocks"
            raise ModelFailed(Failure.PROTOCOL, msg)
        return Reply(
            text=_said(blocks),
            calls=_calls(blocks),
            usage=_usage(document),
            cut=_CUT.get(as_text(as_dict(document).get("stop_reason"))),
        )

    @override
    def explain(self, payload: bytes) -> str:
        try:
            document = decoded(payload, where=self.name)
        except Exception:  # noqa: BLE001 — an error body that is not JSON is not a worse error
            return payload.decode("utf-8", "replace").strip()
        said = as_dict(document).get("error")
        return said.strip() if isinstance(said, str) else as_text(as_dict(said).get("message"))


def _messages(history: tuple[AnyMessage, ...]) -> list[Body]:
    """The conversation, with neighbouring turns of one speaker folded into one message.

    The fold is not a convenience: this dialect takes turns, so two tool results that arrived as two
    canonical messages are one thing it must be told together — and a result followed by a person's line
    is still the person speaking once.
    """
    built: list[Body] = []
    for message in history:
        role, blocks = _spoke(message)
        last = built[-1] if built else None
        if last is not None and last.get("role") == role:
            # ``as_list`` answers the list that is there, so the extend lands on the message already built.
            as_list(last.get("content")).extend(blocks)
            continue
        built.append({"role": role, "content": blocks})
    return built


def _spoke(message: AnyMessage) -> tuple[str, list[Body]]:
    """One canonical message as the speaker it was and the blocks it said."""
    if isinstance(message, User):
        return "user", [{"type": "text", "text": message.text}]
    if isinstance(message, Assistant):
        blocks: list[Body] = [{"type": "text", "text": message.text}] if message.text else []
        blocks.extend(
            {"type": "tool_use", "id": call.id, "name": call.tool, "input": spelled(call.arguments)}
            for call in message.calls
        )
        return "assistant", blocks
    return "user", [_result(message)]


def _result(message: Result) -> Body:
    """One tool's answer as this dialect's ``tool_result`` block, pictures and all.

    ``is_error`` is written because this dialect has the word and the status already says it: a result
    that failed reads as a failure to the model rather than as prose that happens to start with a dash.
    """
    said: Body = {"type": "tool_result", "tool_use_id": message.call, "content": _told(message)}
    if message.status in {Status.REFUSED, Status.FAILED}:
        said["is_error"] = True
    return said


def _told(message: Result) -> str | list[Body]:
    """What a tool's answer carries: its text, or its text and its pictures when there are any.

    Text stays a string when there is nothing beside it, because that is the shorter thing to write and
    the one every provider reads most reliably.
    """
    if not message.pictures:
        return message.text
    blocks: list[Body] = [{"type": "text", "text": message.text}]
    blocks.extend(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": picture.media,
                "data": base64.b64encode(picture.data).decode("ascii"),
            },
        }
        for picture in message.pictures
    )
    return blocks


def _system(prompt: str, caps: Caps) -> str | list[Body]:
    """The prompt that is not the conversation, in the shape the cache mechanism wants.

    A plain string unless a breakpoint is asked for: a marker has to sit *on* a block, and the place
    after the system prompt is the place §10.5 says it belongs — everything before it (the tools, this
    prompt) never moves, and everything after it is the conversation, which does.
    """
    if caps.prefix_cache is not PrefixCache.BREAKPOINT:
        return prompt
    return [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]


def _tools(request: Request, caps: Caps) -> list[Body]:
    """The tool block, with the breakpoint on its last entry when there is no system prompt to hold it."""
    built: list[Body] = [{"name": tool.name, "input_schema": schema(tool.schema)} for tool in request.tools]
    if caps.prefix_cache is PrefixCache.BREAKPOINT and not request.system and built:
        built[-1]["cache_control"] = {"type": "ephemeral"}
    return built


def _said(blocks: Any) -> str:
    """What the answer said: every text block, in order."""
    return "".join(
        as_text(as_dict(block).get("text")) for block in as_list(blocks) if as_dict(block).get("type") == "text"
    )


def _calls(blocks: Any) -> tuple[Call, ...]:
    """The calls the answer asked for, in the order the blocks arrived."""
    return tuple(
        Call(
            id=as_text(block.get("id")),
            tool=as_text(block.get("name")),
            arguments=_spelled(block.get("input")),
        )
        for block in (as_dict(one) for one in as_list(blocks))
        if block.get("type") == "tool_use"
    )


def _spelled(one: Any) -> str:
    """A call's arguments as the text the canonical model keeps them in."""
    return one if isinstance(one, str) else _encoded(one)


def _encoded(one: Any) -> str:
    """Any JSON value as the text it is, which is what a call's arguments are once they arrive."""
    return msgspec.json.encode(one).decode("utf-8")


def _usage(document: Any) -> Usage:
    """What the answer cost, with the cached share this dialect reports by name."""
    usage = as_dict(as_dict(document).get("usage"))
    read = usage.get("cache_read_input_tokens")
    return Usage(
        prompt=as_int(usage.get("input_tokens")),
        completion=as_int(usage.get("output_tokens")),
        cached=None if read is None else as_int(read),
    )
