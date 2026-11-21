"""The chat-completions dialect: one flat list of roles, and the tool calls inside an assistant turn.

The oldest of the shapes and the one every compatible gateway speaks, which is why it is the dialect a
local vLLM, a LiteLLM proxy or a vendor's OpenAI-compatible endpoint is reached over. It is also the
least precise: a conversation is a list of dictionaries keyed by ``role``, a tool call is a member of the
assistant message that asked for it, and a tool's answer is *another* message, since this shape has no
blocks — so "which result answers which call" is carried by an id inside the tool message.

**Two rows, one class.** This wire is spoken by two families that differ in one word: the older one reads
``system`` anywhere in the list, and the newer one reserves it for the front and expects ``developer``
below. So the same codec is filed twice, once per word, and a route picks the family by the name it gives
(``chat-completions`` / ``chat-completions-legacy``, E167) — a name cannot say the wrong word about the
endpoint it was given to, while a configuration key could.

Two things this dialect cannot do, said here so that nobody discovers them later:

* **A picture has no channel.** Its only image slot is inside a *person's* message, and a tool's answer
  is not a person's; putting pixels there would be inventing a speaker. ``pictures`` is therefore
  ``False`` and a route on this dialect is blind however its caps read (see :class:`Codec`).
* **Thinking has two spellings here, and neither is a budget.** An effort level (``reasoning.effort``) is
  what the newer gateways took from ``responses``, and ``chat_template_kwargs.enable_thinking`` is what a
  self-hosted server reads before rendering its own template. A token budget has no portable spelling on
  this wire, so here it is a shape a route cannot ask for (E168). What thinking *comes back* anyway is
  read, kept and handed back (``reasoning_content``), because a vendor that produces it refuses the next
  round without it.

**The one thing it does that the others do not yet**: it reads a streamed answer (``sse``). Each frame
carries a fragment of what the model is saying, and the calls it asks for arrive as fragments keyed by an
index — which is why the calls are assembled at the end, out of the fragments, with no document to read.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm.caps import ToolCalls
from plyngent.kernel.lm.codec import Codec, as_dict, as_int, as_list, as_text, decoded, document, frames, schema
from plyngent.kernel.lm.route import PrefixCache
from plyngent.kernel.lm.thinking import Effort, Off, Template
from plyngent.kernel.model import (
    Assistant,
    Call,
    Delta,
    ModelFailed,
    Note,
    Part,
    Reply,
    Result,
    Thought,
    Usage,
    User,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Any

    from plyngent.kernel.lm.caps import Caps
    from plyngent.kernel.lm.codec import Body
    from plyngent.kernel.lm.thinking import ThinkingSpec
    from plyngent.kernel.model import Request

#: The word a route names the newer of the two rows by.
CHAT = "chat-completions"

#: The word a route names the older family by, where a notice's line says ``system`` (E167).
CHAT_LEGACY = "chat-completions-legacy"

#: The word the last frame of a stream carries: the endpoint's own full stop, and no answer.
DONE = "[DONE]"

#: What a ``finish_reason`` means when it is not "it ended on its own".
#:
#: ``tool_calls`` is deliberately absent: asking for a tool is not being cut off, and an answer that
#: asked for one is complete. ``content_filter`` is ``DENIED`` because the provider decided not to
#: answer, which no retry lifts.
_CUT: dict[str, Failure] = {"length": Failure.LIMIT, "content_filter": Failure.DENIED}


class Chat(Codec):
    """``POST {base}/chat/completions``: messages, tools, and one choice per answer."""

    #: Which of the two rows this instance is: the newer family by default, the older one on request.
    name: str = CHAT
    endpoint: str = "/chat/completions"
    sse: bool = True
    #: The role this row puts a notice's line in, and the one thing the two rows differ by.
    notice: str = ""
    spells: frozenset[type] = frozenset({Off, Effort, Template})

    def __init__(self, *, legacy: bool = False) -> None:
        """One of the two rows: the newer family by default, the older one on request (E167).

        The older family takes the chat template's switch and nothing else, because an effort level is a
        field the newer gateways invented and an endpoint that only knows the older spelling refuses a
        field it does not have (E168).
        """
        self.name = CHAT_LEGACY if legacy else CHAT
        self.notice = "system" if legacy else "developer"
        self.spells = frozenset({Off, Template}) if legacy else frozenset({Off, Effort, Template})

    @override
    def encode(self, request: Request, *, model: str, caps: Caps, cache: PrefixCache) -> Body:
        body: Body = {
            "model": model,
            "messages": [
                *_system(request),
                *(_message(item, self.notice) for item in request.messages),
            ],
            "stream": caps.stream,
            "max_tokens": caps.max_output,
            "parallel_tool_calls": caps.tools is ToolCalls.PARALLEL,
        }
        if request.tools:
            body["tools"] = [
                {"type": "function", "function": {"name": tool.name, "parameters": schema(tool.schema)}}
                for tool in request.tools
            ]
        if cache is PrefixCache.IMPLICIT and request.cache_key:
            body["prompt_cache_key"] = request.cache_key
        thought = _thinking(caps.thinking, self.spells)
        if thought is not None:
            body.update(thought)
        if caps.stream:
            # A streamed answer only carries what it cost when it is asked to: without this the last
            # frame says nothing about tokens, and §10.5's numbers would silently disappear from every
            # streamed turn.
            body["stream_options"] = {"include_usage": True}
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
            thoughts=_thoughts(message),
            usage=_usage(document),
            cut=_CUT.get(as_text(first.get("finish_reason"))),
        )

    @override
    async def stream(self, chunks: AsyncIterator[bytes]) -> AsyncIterator[Delta | Reply]:
        """One streamed answer: each word as it arrives, then the reply the words add up to.

        This dialect streams inside ``choices[0].delta``: a fragment of text, a fragment of a *call* (an
        id once, a name once, and the arguments a few characters at a time), and at the end the reason it
        stopped. So the calls are assembled here: ``index`` says which of them a
        fragment belongs to, and a call is complete only when the stream is.

        ``data: [DONE]`` is the endpoint saying it is finished, and it arrives as a word, not as an event:
        arriving without it is not an error — the body ending is the same thing said differently — but a
        frame that is not JSON is.
        """
        said: list[str] = []
        thought: list[str] = []
        partial: dict[int, _Pending] = {}
        cut: Failure | None = None
        usage = Usage()
        async for _event, data in frames(chunks):
            if data.strip() == DONE:
                break
            frame = as_dict(document(data, where=f"{self.name} stream"))
            usage = _usage(frame) if frame.get("usage") is not None else usage
            choices = as_list(frame.get("choices"))
            delta = as_dict(as_dict(choices[0]).get("delta")) if choices else {}
            text = _said(delta)
            if text:
                said.append(text)
                yield Delta(text=text)
            thinking = as_text(delta.get("reasoning_content"))
            if thinking:
                thought.append(thinking)
                yield Delta(text=thinking, part=Part.THOUGHT)
            _pending(partial, as_list(delta.get("tool_calls")))
            reason = as_text(as_dict(choices[0]).get("finish_reason")) if choices else ""
            if reason:
                cut = _CUT.get(reason)
        yield Reply(text="".join(said), calls=_assembled(partial), thoughts=_kept(thought), usage=usage, cut=cut)

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


class _Pending:
    """One call of a streamed answer, as the fragments of it that have arrived so far."""

    id: str
    tool: str
    arguments: list[str]

    def __init__(self) -> None:
        self.id = ""
        self.tool = ""
        self.arguments = []


def _thinking(thinking: ThinkingSpec, spells: frozenset[type]) -> Body | None:
    """This row's one field for being asked to think, and nothing when it has no word for the shape.

    The declaration is read here as well as where the file is read, so that the two answers cannot drift:
    a shape this row cannot say is not invented into a field the endpoint would refuse, and one it can
    say is exactly one of the two branches below. An effort level is what the newer gateways took from
    ``responses``; ``chat_template_kwargs.enable_thinking`` is what a self-hosted server reads before it
    renders its own template (E168).
    """
    if type(thinking) not in spells:
        return None
    if isinstance(thinking, Effort):
        return {"reasoning": {"effort": thinking.level.value}}
    if isinstance(thinking, Template):
        return {"chat_template_kwargs": {"enable_thinking": thinking.enabled}}
    return None


def _pending(partial: dict[int, _Pending], fragments: list[Any]) -> None:
    """Add the fragments of this frame to the calls they belong to, by the index that names them.

    An index is the only handle a stream gives: the id arrives with the first fragment and never again,
    so a second fragment carrying one would start a *second* call if this keyed on it instead.
    """
    for one in fragments:
        fragment = as_dict(one)
        call = partial.setdefault(as_int(fragment.get("index")), _Pending())
        function = as_dict(fragment.get("function"))
        call.id = as_text(fragment.get("id")) or call.id
        call.tool = as_text(function.get("name")) or call.tool
        call.arguments.append(as_text(function.get("arguments")))


def _assembled(partial: dict[int, _Pending]) -> tuple[Call, ...]:
    """The calls this stream asked for, in the order it asked for them."""
    return tuple(
        Call(id=partial[index].id, tool=partial[index].tool, arguments="".join(partial[index].arguments))
        for index in sorted(partial)
    )


def _system(request: Request) -> list[Body]:
    """The part of the prompt that is not the conversation, as the first message of the list.

    A message here and not a field of its own, because this dialect has no field for one: the parts of
    a prompt that do not move are the front of the list, which is also why they sit before the tools in
    a cached prefix (§10.5).
    """
    return [{"role": "system", "content": request.system}] if request.system else []


def _message(message: User | Assistant | Result | Note, notice: str) -> Body:
    """One canonical message, in the spelling this dialect has for it.

    A notice takes the role this row names, which is the one place in this dialect a notice differs from a
    person's line: this vocabulary has no channel for "the harness", only roles, and a role in the middle
    of the list is exactly what ``system``/``developer`` are for (§10.4, E142).
    """
    if isinstance(message, Note):
        return {"role": notice, "content": message.text}
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
        if message.thoughts:
            # This dialect has one string for a thought and no signature, so what it is handed back is the
            # text it gave. A round that thought nothing gets no field at all: this endpoint is only owed
            # reasoning when it produced some, and asking it for a shape it did not use is noise in a
            # prefix somebody caches (§10.5).
            said["reasoning_content"] = "".join(thought.text for thought in message.thoughts)
        return said
    return {"role": "tool", "tool_call_id": message.call, "content": message.text}


def _thoughts(message: Any) -> tuple[Thought, ...]:
    """What one answer thought, in this dialect's one string for it."""
    return _kept([as_text(as_dict(message).get("reasoning_content"))])


def _kept(thought: list[str]) -> tuple[Thought, ...]:
    """The thinking a stream gathered, as the one thought this dialect spells it as."""
    said = "".join(thought)
    return (Thought(text=said),) if said else ()


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
    numbers are the point of §10.5, so both spellings are read here, since a report left to itself would
    silently say zero.
    """
    usage = as_dict(as_dict(document).get("usage"))
    details = as_dict(usage.get("prompt_tokens_details"))
    cached = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
    return Usage(
        prompt=as_int(usage.get("prompt_tokens")),
        completion=as_int(usage.get("completion_tokens")),
        cached=None if cached is None else as_int(cached),
    )
