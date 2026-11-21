"""The responses dialect: an ``instructions`` field, and a conversation as a list of typed items.

The third row, and the one that proves the shape: adding it changed this file and one line of the table,
and nothing else — no loop, no journal, no cap, no command. Its conversation is a list of *items*:
a message item, a ``function_call`` item, a ``function_call_output`` item, each standing
on its own, which is why a run of results needs no merging here the way it does in ``messages``.

Two things it does differently from the other two, both of them this endpoint's opinion:

* **The prompt that is not the conversation is a field** (``instructions``) of its own, with no leading
  message, so a cached prefix starts at it and the tools follow.
* **A picture travels inside a tool's answer** as an ``input_image`` part beside the ``input_text`` one,
  so ``pictures`` is ``True`` and a route on this dialect may honestly claim ``Caps.vision``.

Thinking is written as an effort level from a closed set, which is this endpoint's own word for it, and
not the token count ``messages`` takes (E168). What an answer thinks anyway arrives as a ``reasoning``
item in front of what it said, and that item goes back with the round: this endpoint refuses a request
that carries a tool round's calls without the reasoning they came with.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, override

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm.caps import PrefixCache
from plyngent.kernel.lm.codec import Codec, as_dict, as_int, as_list, as_text, decoded, schema
from plyngent.kernel.lm.thinking import Effort, Off
from plyngent.kernel.model import Assistant, Call, ModelFailed, Note, Reply, Result, Thought, Usage, User

if TYPE_CHECKING:
    from typing import Any

    from plyngent.kernel.lm.caps import Caps
    from plyngent.kernel.lm.codec import Body
    from plyngent.kernel.model import AnyMessage, Request

#: The word a route names this dialect by.
RESPONSES = "responses"

#: What an ``incomplete_details.reason`` means when the answer did not simply end.
#:
#: ``content_filter`` is ``DENIED`` for the same reason ``refusal`` is in ``messages``: the provider
#: decided not to answer, and no retry lifts that.
_CUT: dict[str, Failure] = {"max_output_tokens": Failure.LIMIT, "content_filter": Failure.DENIED}


class Responses(Codec):
    """``POST {base}/responses``: instructions, typed items, and ``max_output_tokens``."""

    name: str = RESPONSES
    endpoint: str = "/responses"
    pictures: bool = True
    spells: frozenset[type] = frozenset({Off, Effort})

    @override
    def encode(self, request: Request, *, model: str, caps: Caps) -> Body:
        body: Body = {
            "model": model,
            "input": [item for message in request.messages for item in _items(message)],
            # Not one this dialect can read yet (``sse`` is False), so it never asks for one.
            "stream": False,
            "max_output_tokens": caps.max_output,
            "parallel_tool_calls": caps.parallel_tools,
        }
        if request.system:
            body["instructions"] = request.system
        if request.tools:
            body["tools"] = [
                {"type": "function", "name": tool.name, "parameters": schema(tool.schema)} for tool in request.tools
            ]
        if caps.prefix_cache is PrefixCache.IMPLICIT and request.cache_key:
            body["prompt_cache_key"] = request.cache_key
        if isinstance(caps.thinking, Effort):
            body["reasoning"] = {"effort": caps.thinking.level.value}
        return body

    @override
    def decode(self, payload: bytes) -> Reply:
        document = decoded(payload, where=self.name)
        output = as_list(as_dict(document).get("output"))
        if not output:
            msg = f"{self.name} answered with no items, so nothing was said"
            raise ModelFailed(Failure.PROTOCOL, msg)
        return Reply(
            text=_said(output),
            calls=_calls(output),
            thoughts=_thoughts(output),
            usage=_usage(document),
            cut=_CUT.get(as_text(as_dict(as_dict(document).get("incomplete_details")).get("reason"))),
        )

    @override
    def explain(self, payload: bytes) -> str:
        try:
            document = decoded(payload, where=self.name)
        except Exception:  # noqa: BLE001 — an error body that is not JSON is not a worse error
            return payload.decode("utf-8", "replace").strip()
        said = as_dict(document).get("error")
        return said.strip() if isinstance(said, str) else as_text(as_dict(said).get("message"))


def _items(message: AnyMessage) -> list[Body]:
    """One canonical message as the one item this dialect has for it.

    One item each, unlike ``messages``: an answer that asked for two tools becomes an answer item and two
    ``function_call`` items, and each of them is addressed by its own ``call_id`` — so nothing has to be
    folded together at the front of the next turn. A notice is a ``developer`` message and needs no cap to
    say so: this vocabulary has that role and no other for a line the harness says (§10.4, E142).
    """
    if isinstance(message, User):
        return [{"role": "user", "content": [{"type": "input_text", "text": message.text}]}]
    if isinstance(message, Note):
        return [{"role": "developer", "content": [{"type": "input_text", "text": message.text}]}]
    if isinstance(message, Assistant):
        said: list[Body] = []
        thinking = _thinked(message.thoughts)
        if thinking is not None:
            said.append(thinking)
        if message.text:
            said.append({"role": "assistant", "content": [{"type": "output_text", "text": message.text}]})
        said.extend(
            {"type": "function_call", "call_id": call.id, "name": call.tool, "arguments": call.arguments or "{}"}
            for call in message.calls
        )
        return said
    return [_output(message)]


def _thinked(thoughts: tuple[Thought, ...]) -> Body | None:
    """One round's thinking as the item this dialect reads it back from, or nothing when it had none.

    Not an empty item when the thinking was empty: this endpoint refuses a reasoning item with no part in
    it, and a round that thought nothing is a round that owes nothing.
    """
    parts = [{"type": "reasoning_text", "text": thought.text} for thought in thoughts if thought.text]
    return {"type": "reasoning", "content": parts} if parts else None


def _output(message: Result) -> Body:
    """One tool's answer, with its pictures when there are any."""
    return {"type": "function_call_output", "call_id": message.call, "output": _told(message)}


def _told(message: Result) -> str | list[Body]:
    """What a tool's answer carries: its text, or its text and its pictures when there are any."""
    if not message.pictures:
        return message.text
    told: list[Body] = [{"type": "input_text", "text": message.text}]
    told.extend(
        {
            "type": "input_image",
            "image_url": f"data:{picture.media};base64,{base64.b64encode(picture.data).decode('ascii')}",
        }
        for picture in message.pictures
    )
    return told


def _said(output: list[Any]) -> str:
    """What the answer said: every output text of every message item, in order."""
    said: list[str] = []
    for item in output:
        if as_dict(item).get("type") != "message":
            continue
        said.extend(
            as_text(as_dict(part).get("text"))
            for part in as_list(as_dict(item).get("content"))
            if as_dict(part).get("type") == "output_text"
        )
    return "".join(said)


def _thoughts(output: list[Any]) -> tuple[Thought, ...]:
    """What the answer thought, as the reasoning items it put in front of what it said."""
    return tuple(
        Thought(text=as_text(part.get("text")))
        for item in (as_dict(one) for one in output)
        if item.get("type") == "reasoning"
        for part in (as_dict(one) for one in as_list(item.get("content")))
        if part.get("type") == "reasoning_text" and as_text(part.get("text"))
    )


def _calls(output: list[Any]) -> tuple[Call, ...]:
    """The calls the answer asked for, in the order the items arrived."""
    return tuple(
        Call(
            id=as_text(item.get("call_id")),
            tool=as_text(item.get("name")),
            arguments=as_text(item.get("arguments")),
        )
        for item in (as_dict(one) for one in output)
        if item.get("type") == "function_call"
    )


def _usage(document: Any) -> Usage:
    """What the answer cost, with the cached share this dialect nests one level down."""
    usage = as_dict(as_dict(document).get("usage"))
    details = as_dict(usage.get("input_tokens_details"))
    cached = details.get("cached_tokens")
    return Usage(
        prompt=as_int(usage.get("input_tokens")),
        completion=as_int(usage.get("output_tokens")),
        cached=None if cached is None else as_int(cached),
    )
