"""The dialects, as pure functions: a request becomes a body, a body becomes a reply.

A codec has no socket and no clock, so what it does is exactly what these tests do — hand it values and
read the values it answers. That is also why they are here: the four
faces a story stands outside of are a screen, a request, an answer and a disk, and the *body* between
the second and the third is the one thing a story cannot see without a server. Every claim below is a
claim about bytes, and the stories about routes then assert the same bytes arrive at a real server.

Kept to the properties the two dialects must agree about and the places they must not. A dialect's
whole content is "where does this go in *my* JSON", and a test that repeated the JSON back would pin the
spelling, not the rule.
"""

from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, cast

import msgspec
import pytest

from plyngent.kernel.failure import Failure
from plyngent.kernel.lm import (
    DIALECTS,
    Caps,
    Chat,
    Codec,
    Messages,
    PrefixCache,
    Responses,
    codec_named,
)
from plyngent.kernel.lm.codec import of_status, refused
from plyngent.kernel.model import (
    Assistant,
    Call,
    ModelFailed,
    Note,
    Picture,
    Request,
    Result,
    Thought,
    ToolDecl,
    User,
)
from plyngent.kernel.toolresult import Status
from plyngent.kernel.trail import NOTICE_MARKER

if TYPE_CHECKING:
    from typing import Any

#: A conversation with one of everything: a person's line, a call, and what the call came back with.
CONVERSATION: tuple[User | Assistant | Result, ...] = (
    User(text="what is in this directory?"),
    Assistant(text="looking", calls=(Call(id="c1", tool="listdir", arguments='{"path":"."}'),)),
    Result(call="c1", tool="listdir", text="a.txt\nb.txt", documents=("d1",)),
)

#: The prompt that is not the conversation — one line, which is all this build assembles so far.
SYSTEM = "you may not reach: net"

#: One tool declaration, as the canonical model keeps it: the schema is JSON text.
TOOL = ToolDecl(name="listdir", schema=json.dumps({"type": "object", "properties": {"path": {"type": "string"}}}))


def _request(*, messages: tuple[User | Assistant | Result, ...] = CONVERSATION, system: str = SYSTEM) -> Request:
    """The conversation above, as a request a turn would assemble."""
    return Request(
        system=system,
        messages=messages,
        tools=(TOOL,),
        route="local",
        cache_key="main:1:0",
    )


def _caps(**given: object) -> Caps:
    """The conservative route: no vision, one call per answer, a stated ceiling."""
    return Caps(max_output=1024, **given)  # type: ignore[arg-type]


def _body(codec: Codec, request: Request, *, caps: Caps) -> dict[str, Any]:
    """One request as the JSON a server would receive, read back into something a test can walk.

    Through the bytes on purpose: what a dialect promises is a *document*, and the text it becomes is
    what a server actually gets. Reading it back also keeps the assertions about the wire, not
    about the implementation of the encoder.
    """
    written = msgspec.json.encode(codec.encode(request, model="m", caps=caps))
    return cast("dict[str, Any]", msgspec.json.decode(written))


def test_the_two_dialects_are_the_same_conversation() -> None:
    """The same conversation reaches both endpoints: a system line, a call, and an answer to it.

    What differs is only where each of those lives — the prompt is a message in one dialect and a field
    in the other; a call is a member of the assistant message in one and a block inside it in the other;
    a result is a message of its own with a role, or a block inside the person's next turn.
    """
    request = _request()

    chat = _body(Chat(), request, caps=_caps())
    messages = _body(Messages(), request, caps=_caps())

    said = chat["messages"]
    assert isinstance(said, list)
    assert said[0] == {"role": "system", "content": "you may not reach: net"}
    assert said[1] == {"role": "user", "content": "what is in this directory?"}
    assert said[2]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "listdir", "arguments": '{"path":"."}'}}
    ]
    assert said[3] == {"role": "tool", "tool_call_id": "c1", "content": "a.txt\nb.txt"}

    assert messages["system"] == "you may not reach: net"
    spoken = messages["messages"]
    assert spoken[0] == {"role": "user", "content": [{"type": "text", "text": "what is in this directory?"}]}
    assert spoken[1]["content"][1] == {"type": "tool_use", "id": "c1", "name": "listdir", "input": {"path": "."}}
    assert spoken[2]["content"] == [{"type": "tool_result", "tool_use_id": "c1", "content": "a.txt\nb.txt"}]


def test_a_notice_is_put_where_each_dialect_has_room_for_it() -> None:
    """E142: the harness speaking takes a role, a developer item, and the person's role — in that order.

    Three answers because the three vocabularies differ: a flat list of roles has a word for "instructions
    below the front", typed items have ``developer``, and the block dialect has no role for it at all — so
    there the notice is a block of the person's turn, and the marker in its text is what keeps the two
    apart.
    """
    note = Note(text=f"{NOTICE_MARKER}\n2 things are still on your list")
    request = _request(messages=(User(text="hi"), note))

    chat = _body(Chat(), request, caps=_caps())["messages"]
    assert chat[2] == {"role": "developer", "content": note.text}

    spoken = _body(Messages(), request, caps=_caps())["messages"]
    assert [block["text"] for block in spoken[0]["content"]] == ["hi", note.text], (
        "this dialect takes turns, so the notice is a block of the turn it was said in"
    )

    items = _body(Responses(), request, caps=_caps())["input"]
    assert items[1] == {"role": "developer", "content": [{"type": "input_text", "text": note.text}]}


def test_the_older_row_puts_a_notice_in_the_role_it_is_named_for() -> None:
    """E167: the two chat rows differ by one word, and the word is the row's name.

    ``chat-completions`` is the newer family, which reserves ``system`` for the front of the list; the
    older one reads ``system`` anywhere. Both are the same codec filed twice, so this is a test of the
    constructor and not of a second spelling of the wire.
    """
    note = Note(text=f"{NOTICE_MARKER}\n2 things are still on your list")
    request = _request(messages=(User(text="hi"), note))

    newer = _body(Chat(), request, caps=_caps())["messages"]
    older = _body(Chat(legacy=True), request, caps=_caps())["messages"]

    assert newer[2] == {"role": "developer", "content": note.text}
    assert older[2] == {"role": "system", "content": note.text}


def test_a_run_of_results_is_one_turn_of_the_person_speaking() -> None:
    """Two calls answered are one message in the dialect that takes turns, and two in the one that does not.

    This is the merge that only one dialect needs, and it is a property of *that* dialect: a model asked
    to take turns is told about both results at once, which is also why the fold has to be here, and not
    in whoever built the conversation.
    """
    request = _request(
        messages=(
            User(text="two files please"),
            Assistant(text="", calls=(Call(id="c1", tool="read"), Call(id="c2", tool="read"))),
            Result(call="c1", tool="read", text="one"),
            Result(call="c2", tool="read", text="two"),
        )
    )

    spoken = _body(Messages(), request, caps=_caps())["messages"]
    assert [message["role"] for message in spoken] == ["user", "assistant", "user"]
    assert [block["type"] for block in spoken[2]["content"]] == ["tool_result", "tool_result"]
    assert [block["tool_use_id"] for block in spoken[2]["content"]] == ["c1", "c2"]

    flat = _body(Chat(), request, caps=_caps())["messages"]
    assert [said["role"] for said in flat[1:]] == ["user", "assistant", "tool", "tool"]


def test_a_failed_answer_reads_as_a_failure_where_the_dialect_has_the_word() -> None:
    """A refused result is marked as one, so the model does not read a failure as prose.

    The status is already the kernel's answer to "how did this go"; a dialect that can carry it carries
    it, and one that cannot says the same thing in the summary the model reads anyway.
    """
    request = _request(
        messages=(
            Assistant(text="", calls=(Call(id="c1", tool="read"),)),
            Result(call="c1", tool="read", text="read: no", status=Status.REFUSED),
        )
    )

    spoken = _body(Messages(), request, caps=_caps())["messages"]
    assert spoken[0]["content"] == [{"type": "tool_use", "id": "c1", "name": "read", "input": {}}]
    assert spoken[1]["content"][0]["is_error"] is True
    assert _body(Chat(), request, caps=_caps())["messages"][2]["role"] == "tool"


def test_a_prefix_key_travels_only_where_the_route_asks_for_one() -> None:
    """Three mechanisms, three bodies: a key, a marker, or nothing at all (§10.5).

    The marker has to sit *after* the parts that never move — the tools and the system prompt — which is
    why it lands on the system block, and on the last tool when there is no system prompt to carry it.
    """
    request = _request()

    silent = _body(Messages(), request, caps=_caps())
    marked = _body(Messages(), request, caps=_caps(prefix_cache=PrefixCache.BREAKPOINT))
    keyed = _body(Chat(), request, caps=_caps(prefix_cache=PrefixCache.IMPLICIT))

    assert silent["system"] == "you may not reach: net"
    assert marked["system"] == [
        {"type": "text", "text": "you may not reach: net", "cache_control": {"type": "ephemeral"}}
    ]
    assert keyed["prompt_cache_key"] == "main:1:0"
    assert "prompt_cache_key" not in _body(Chat(), request, caps=_caps())

    no_prompt = _request(system="")
    tools_marked = _body(Messages(), no_prompt, caps=_caps(prefix_cache=PrefixCache.BREAKPOINT))
    assert tools_marked["tools"][-1]["cache_control"] == {"type": "ephemeral"}


def test_the_same_request_is_the_same_bytes_twice() -> None:
    """A cached prefix is worth what the determinism of what precedes it is (§10.5).

    Two encodings of one value, byte for byte — the property the whole cache story rests on, asserted
    where it can be seen: at the two functions that produce the bytes.
    """
    for codec in DIALECTS:
        once = msgspec.json.encode(codec.encode(_request(), model="m", caps=_caps()))
        twice = msgspec.json.encode(codec.encode(_request(), model="m", caps=_caps()))
        assert once == twice, f"{codec.name} wrote two different bodies for one request"


def test_an_answer_that_ran_out_of_room_says_which_limit_it_hit() -> None:
    """Both dialects name being cut off in the kernel's own closed set, not in their own words."""
    cut = Chat().decode(b'{"choices":[{"message":{"content":"half a sen"},"finish_reason":"length"}],"usage":{}}')
    filtered = Messages().decode(b'{"content":[{"type":"text","text":""}],"stop_reason":"max_tokens","usage":{}}')
    ended = Messages().decode(b'{"content":[{"type":"text","text":"done"}],"stop_reason":"end_turn","usage":{}}')

    assert cut.cut is Failure.LIMIT
    assert filtered.cut is Failure.LIMIT
    assert ended.cut is None
    assert ended.text == "done"


def test_what_an_answer_asked_for_arrives_as_calls() -> None:
    """A call is the same three things in both dialects, whichever way the answer spelled it."""
    chart = Chat().decode(
        b'{"choices":[{"message":{"content":null,"tool_calls":[{"id":"a","function":{"name":"read",'
        b'"arguments":"{\\"path\\":\\"x\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{}}'
    )
    spoken = Messages().decode(
        b'{"content":[{"type":"tool_use","id":"a","name":"read","input":{"path":"x"}}],'
        b'"stop_reason":"tool_use","usage":{}}'
    )

    for reply in (chart, spoken):
        assert reply.calls == (Call(id="a", tool="read", arguments='{"path":"x"}'),)
        assert reply.cut is None


def test_a_thought_is_given_back_with_the_round_in_each_dialects_own_spelling() -> None:
    """E145: the thinking is not a display, it is part of the next request.

    A thinking endpoint refuses a tool round whose thinking went missing, so what is asserted is the
    body the *following* round would send. The block dialect is the strict one — it needs the text and
    the signature the endpoint signed it with — and what a provider redacted is ciphertext that has to
    go back byte for byte. ``chat`` has one string and no signature, and invents neither.
    """
    thoughts = (Thought(text="look first", signature="s1"), Thought(sealed="cipher"))
    answer = Assistant(
        text="calling",
        calls=(Call(id="c1", tool="listdir", arguments='{"path":"."}'),),
        thoughts=thoughts,
    )
    request = _request(messages=(User(text="hi"), answer))

    spoken = _body(Messages(), request, caps=_caps())["messages"][1]["content"]
    assert spoken[:2] == [
        {"type": "thinking", "thinking": "look first", "signature": "s1"},
        {"type": "redacted_thinking", "data": "cipher"},
    ], "the thinking is a block of the same message, and it comes before what it led to"
    assert spoken[2] == {"type": "text", "text": "calling"}

    chat = _body(Chat(), request, caps=_caps())["messages"][2]
    assert chat["reasoning_content"] == "look first", "one string, and the part this build cannot spell is not invented"

    items = _body(Responses(), request, caps=_caps())["input"]
    assert items[1] == {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "look first"}]}, (
        "the reasoning item stands in front of what the round said, which is where the endpoint reads it"
    )
    assert items[2]["role"] == "assistant", "and what the round said follows it, in the order it was said"


def test_an_answer_that_thought_is_read_as_the_thought_it_was() -> None:
    """The other half: three ways of saying it, read into the one value the next round writes back."""
    chat = Chat().decode(
        b'{"choices":[{"message":{"role":"assistant","content":"ok","reasoning_content":"hmm"},'
        b'"finish_reason":"stop"}],"usage":{}}'
    )
    spoken = Messages().decode(
        b'{"content":[{"type":"thinking","thinking":"hmm","signature":"s1"},'
        b'{"type":"redacted_thinking","data":"cipher"},{"type":"text","text":"ok"}],'
        b'"stop_reason":"end_turn","usage":{}}'
    )
    items = Responses().decode(
        b'{"output":[{"type":"reasoning","content":[{"type":"reasoning_text","text":"hmm"}]},'
        b'{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}],"usage":{}}'
    )

    assert chat.thoughts == (Thought(text="hmm"),), "this dialect signs nothing, so nothing is invented"
    assert spoken.thoughts == (Thought(text="hmm", signature="s1"), Thought(sealed="cipher")), (
        "an unreadable block is kept as what it is, not dropped for being unreadable"
    )
    assert items.thoughts == (Thought(text="hmm"),)
    assert chat.text == spoken.text == items.text == "ok", "and the answer beside it is untouched"


def test_an_answer_that_thought_nothing_owes_nothing() -> None:
    """A round with no thinking writes no field: an empty one is a shape the endpoint did not use."""
    request = _request(
        messages=(User(text="hi"), Assistant(text="calling", calls=(Call(id="c1", tool="go", arguments="{}"),)))
    )

    assert "reasoning_content" not in _body(Chat(), request, caps=_caps())["messages"][2]
    assert [item.get("type") for item in _body(Responses(), request, caps=_caps())["input"]] == [
        None,
        None,
        "function_call",
    ], "a person's line, the answer, and the call it asked for — and no reasoning item at all"
    assert _body(Messages(), request, caps=_caps())["messages"][1]["content"] == [
        {"type": "text", "text": "calling"},
        {"type": "tool_use", "id": "c1", "name": "go", "input": {}},
    ]


def test_the_cached_share_is_read_in_whichever_spelling_the_endpoint_reports() -> None:
    """Three spellings of "this much was already there", and silence kept apart from zero.

    §10.5 exists so that "the prefix was cheaper" is a countable claim. A dialect that reads one spelling
    and not the other would report zero for a whole vendor, which is the one answer that looks true.
    """
    nested = Chat().decode(
        b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_tokens_details":{"cached_tokens":8}}}'
    )
    flat = Chat().decode(
        b'{"choices":[{"message":{"content":"x"},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":2,"prompt_cache_hit_tokens":8}}'
    )
    messages = Messages().decode(
        b'{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",'
        b'"usage":{"input_tokens":10,"output_tokens":2,"cache_read_input_tokens":8}}'
    )
    silent = Messages().decode(
        b'{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn",'
        b'"usage":{"input_tokens":10,"output_tokens":2}}'
    )

    assert (nested.usage.prompt, nested.usage.completion, nested.usage.cached) == (10, 2, 8)
    assert flat.usage.cached == 8
    assert messages.usage.cached == 8
    assert silent.usage.cached is None


def test_a_picture_has_a_channel_only_where_a_dialect_has_one() -> None:
    """Pixels travel as bytes on the dialect that can carry them, and are not invented on the one that cannot.

    This is why :attr:`plyngent.kernel.lm.codec.Codec.pictures` exists: a route's ``vision`` cap says what
    its *model* can read, and this says what its *dialect* can carry, and a route needs both.
    """
    picture = Picture(media="image/png", data=b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    request = _request(
        messages=(
            Assistant(text="", calls=(Call(id="c1", tool="read"),)),
            Result(call="c1", tool="read", text="kept as d1", documents=("d1",), pictures=(picture,)),
        )
    )

    spoken = _body(Messages(), request, caps=_caps(vision=True))["messages"]
    blocks = spoken[1]["content"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "kept as d1"}
    assert blocks[1]["type"] == "image"
    assert blocks[1]["source"]["media_type"] == "image/png"
    assert base64.b64decode(blocks[1]["source"]["data"]) == picture.data

    assert Chat.pictures is False
    assert Messages.pictures is True


def test_another_dialect_is_one_more_file_and_one_more_row() -> None:
    """The third dialect changed one module and one line of the table, and that is the whole claim.

    This is §6's falsifiable slice for R11 written as a test: the conversation is the same three things,
    and only their spelling in *this* endpoint's JSON differs — an ``instructions`` field, typed items,
    and a call addressed by ``call_id``.
    """
    request = _request()
    body = _body(Responses(), request, caps=_caps(prefix_cache=PrefixCache.IMPLICIT))

    assert body["instructions"] == SYSTEM
    assert body["model"] == "m"
    assert body["max_output_tokens"] == 1024
    assert body["prompt_cache_key"] == "main:1:0"
    assert body["tools"] == [
        {"type": "function", "name": "listdir", "parameters": json.loads(TOOL.schema)},
    ]
    assert body["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "what is in this directory?"}],
    }
    assert body["input"][1] == {"role": "assistant", "content": [{"type": "output_text", "text": "looking"}]}
    assert body["input"][2] == {
        "type": "function_call",
        "call_id": "c1",
        "name": "listdir",
        "arguments": '{"path":"."}',
    }
    assert body["input"][3] == {"type": "function_call_output", "call_id": "c1", "output": "a.txt\nb.txt"}

    answer = Responses().decode(
        b'{"output":[{"type":"message","content":[{"type":"output_text","text":"four files"}]},'
        b'{"type":"function_call","call_id":"c2","name":"read","arguments":"{\\"path\\":\\"a\\"}"}],'
        b'"status":"completed","usage":{"input_tokens":9,"output_tokens":4,'
        b'"input_tokens_details":{"cached_tokens":7}}}'
    )

    assert answer.text == "four files"
    assert answer.calls == (Call(id="c2", tool="read", arguments='{"path":"a"}'),)
    assert answer.usage.cached == 7
    assert answer.cut is None


def test_a_route_names_its_dialect_and_a_dialect_it_does_not_have_is_not_a_name() -> None:
    """The table is one row per dialect, looked up by the word a route carries."""
    assert codec_named("messages") is not None
    assert codec_named("chat-completions") is not None
    assert codec_named("chat-completions-legacy") is not None
    assert codec_named("responses") is not None
    assert codec_named("carrier-pigeon") is None
    assert [codec.name for codec in DIALECTS] == [
        "messages",
        "chat-completions",
        "chat-completions-legacy",
        "responses",
    ]


def test_a_refused_request_keeps_the_status_kind_and_the_endpoints_words() -> None:
    """One status→kind table for every dialect, and the sentence the endpoint actually said.

    ``429`` is ``BUSY`` and ``401`` is ``PERMISSION`` — that is what those numbers mean for every
    endpoint there is, whatever a dialect says; and the sentence is the endpoint's own,
    because a sentence invented here would be a guess about somebody else's reason.
    """
    body = b'{"error": {"type": "rate_limit", "message": "slow down"}}'

    busy = refused(Messages(), body, status=429, where="messages")
    denied = refused(Chat(), b'{"error":{"message":"no key"}}', status=401, where="chat-completions")

    assert isinstance(busy, ModelFailed)
    assert busy.failure is Failure.BUSY
    assert str(busy) == "messages answered 429: slow down"
    assert denied.failure is Failure.PERMISSION
    assert of_status(503) is Failure.BUSY
    assert of_status(413) is Failure.LIMIT
    assert of_status(418) is Failure.PROTOCOL


def test_a_body_that_is_not_an_answer_is_a_protocol_failure() -> None:
    """Nothing here guesses: a body with no choices, or bytes that are not JSON, is a ``PROTOCOL``."""
    with pytest.raises(ModelFailed) as nothing:
        _ = Chat().decode(b'{"choices": []}')
    with pytest.raises(ModelFailed) as garbage:
        _ = Messages().decode(b"<html>")
    with pytest.raises(ModelFailed) as wrong:
        _ = Messages().decode(b'{"content":"not a list"}')

    assert all(found.value.failure is Failure.PROTOCOL for found in (nothing, garbage, wrong))
