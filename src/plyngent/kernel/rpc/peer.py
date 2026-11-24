"""A conversation in JSON-RPC over a line: who is waiting on what, and what happens when it stops.

A peer is one end of a talk two processes have, and the four rules below are the whole of it. What it
does *not* decide is as important: no method name is known here, no deadline is invented, and nothing
is retried — the thing above this module knows what it asked for and what a refusal means.

* **An answer belongs to the id it was asked with.** Several requests may be in flight and their
  answers may come back in any order, so a table of ids and the futures waiting on them is the whole of
  the pairing. An id nobody is waiting for is dropped: the call it belonged to was given up on, and an
  answer arriving afterwards cannot change that.
* **A notification is handed over and not waited for.** Handlers are plain functions, because that is
  all this tree asks of one — "the list of tools changed" is a flag and not a task — and a handler that
  raised must not look like the other end going quiet.
* **A request arriving from the other end is answered by what this end offers.** A peer with no
  :class:`~plyngent.kernel.rpc.serving.Serving` offers nothing, and says "no such method" — silence
  would leave the other end waiting forever.
* **Everything ends together.** When the reading stops — the stream ended, a line was not a message —
  every caller still waiting is failed, because the conversation they were waiting on is over, and
  telling one of them about it is not truer than telling all of them.
"""

from __future__ import annotations

import asyncio
import contextlib
from itertools import count
from typing import TYPE_CHECKING, cast

from plyngent.kernel.rpc.framing import Lines, decode, encode

if TYPE_CHECKING:
    from asyncio import StreamWriter
    from collections.abc import Callable, Mapping
    from typing import Any

    from plyngent.kernel.rpc.serving import Serving

#: The number JSON-RPC gives "there is no such method", which is the one answer this end has for a
#: request nobody here has a handler for: a peer with no table offers nothing at all.
METHOD_NOT_FOUND = -32601

#: What a notification is handed: its params, and nothing else — a handler is told what happened, not
#: which conversation it happened in.
type Notice = Callable[[dict[str, Any]], None]


class RpcError(Exception):
    """The other end answered, and the answer was a refusal.

    ``code`` and ``data`` are carried unchanged, because which numbers mean "you asked wrongly" and
    which mean "I am unwell" is the protocol's business: this module refuses to have an opinion about
    a vocabulary it does not speak.
    """

    code: int
    message: str
    data: Any

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class PeerGone(Exception):  # noqa: N818 — the conversation ending is a state, not a fault of the call
    """The conversation is over: the stream ended, a write found nothing to write into, or it was
    closed by this end on purpose.

    It is one exception for all three because the answer to all three is the same — nothing more can be
    said over this peer, and whatever was waiting has stopped waiting — and a caller that had to tell
    them apart would be a caller deciding whether to restart from an exception's class.
    """


class ProtocolBroken(Exception):  # noqa: N818 — what the other end is, not a failed call of ours
    """The other end stopped speaking the protocol: a line that is not a message, an answer that is
    neither a result nor an error, or an answer to an id this end never sent.

    Kept apart from :class:`PeerGone` because it says something about the program on the other side:
    it was not built to be spoken to, or it is not the program it was thought to be.
    """


class Peer:
    """One end of a JSON-RPC conversation, over a stream somebody else opened.

    The stream is not opened and not closed here: a peer is handed two ends of a child somebody else
    started, so "how the process was made" and "how it was spoken to" stay two answers. Closing the
    peer stops the talking; killing the child is the caller's, and so is starting another one.

    ``notices`` is fixed at construction because what a peer acts on is a fact about the protocol above
    it: a handler registered halfway through a conversation would be a method whose notifications were
    sometimes read and sometimes dropped. ``serving`` is fixed for the same reason, and is the other
    direction of one protocol: the methods the other end may call here.
    """

    _lines: Lines
    _writer: StreamWriter
    _notices: Mapping[str, Notice]
    _serving: Serving | None
    _ids: count[int]
    _waiting: dict[int, asyncio.Future[Any]]
    _answering: set[asyncio.Task[None]]
    _out: asyncio.Queue[bytes | None]
    _reading: asyncio.Task[None] | None
    _writing: asyncio.Task[None] | None
    _ended: PeerGone | ProtocolBroken | None

    def __init__(
        self,
        lines: Lines,
        writer: StreamWriter,
        *,
        notices: Mapping[str, Notice] | None = None,
        serving: Serving | None = None,
    ) -> None:
        self._lines = lines
        self._writer = writer
        self._notices = dict(notices or {})
        self._serving = serving
        self._ids = count(1)
        self._waiting = {}
        self._answering = set()
        self._out = asyncio.Queue()
        self._reading = None
        self._writing = None
        self._ended = None

    @property
    def failure(self) -> PeerGone | ProtocolBroken | None:
        """Why this conversation is over, or ``None`` while it still is one."""
        return self._ended

    def start(self) -> None:
        """Begin reading and writing, once. Nothing is sent before this."""
        self._reading = asyncio.create_task(self._read())
        self._writing = asyncio.create_task(self._write())

    async def call(
        self, method: str, params: Mapping[str, Any] | None = None, *, timeout_seconds: float | None = None
    ) -> Any:
        """Ask for one method, and answer what it returned.

        The id is given up on the way out when the deadline passes or the caller is cancelled, so a
        late answer finds nobody waiting and is dropped — the same rule as an answer that arrives after
        the turn it belonged to was stopped. ``timeout_seconds`` bounds the answer and says nothing
        about the method: what a deadline means for a call is the caller's to say.
        """
        self._speaking()
        number = next(self._ids)
        waiting: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._waiting[number] = waiting
        try:
            self._say({"jsonrpc": "2.0", "id": number, "method": method, "params": dict(params or {})})
            async with asyncio.timeout(timeout_seconds):
                return await waiting
        finally:
            _ = self._waiting.pop(number, None)

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        """Say something nobody answers — including, when the conversation is over, nothing at all.

        A notification has no caller waiting on it, so there is nobody to hand a failure to: the fact
        that it was not said is the same fact as the peer being gone, which its reader has already been
        told by whatever call failed.
        """
        if self._ended is not None:
            return
        self._say({"jsonrpc": "2.0", "method": method, "params": dict(params or {})})

    async def close(self) -> None:
        """Stop talking, on the way out of a process that is about to end anyway."""
        self._end(PeerGone("the conversation was closed"))
        for task in self._answering:
            _ = task.cancel()
        for task in (self._reading, self._writing):
            if task is None:
                continue
            _ = task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._answering.clear()
        self._reading = None
        self._writing = None

    # -- what is written -----------------------------------------------------------

    def _speaking(self) -> None:
        """Refuse a call when there is no conversation left to ask it in."""
        if self._ended is not None:
            raise self._ended

    def _say(self, message: Mapping[str, Any]) -> None:
        """Hand one message to the writer, which is the only thing that touches the stream."""
        self._out.put_nowait(encode(message))

    async def _write(self) -> None:
        """Everything this end says, in the order it was said, until there is nothing more.

        One task and a queue, with no lock around a write: the order messages leave in is then
        the order they were queued in, which is what a protocol with ids is allowed to assume, and no
        caller can be interleaved into the middle of another one's bytes.
        """
        try:
            while (outgoing := await self._out.get()) is not None:
                self._writer.write(outgoing)
                await self._writer.drain()
        except asyncio.CancelledError:
            raise
        except OSError as exc:
            self._end(PeerGone(f"could not write: {exc}"))

    # -- what is read --------------------------------------------------------------

    async def _read(self) -> None:
        """Every message the other end sends, until there are none.

        A line that cannot be read ends the conversation and is never skipped: nothing here can
        tell a program that printed a debug line from one that is answering a different protocol, and
        carrying on would put an answer to the wrong question into somebody's result.
        """
        try:
            while (line := await self._lines.line()) is not None:
                self._took(decode(line))
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError) as exc:
            self._end(ProtocolBroken(str(exc)))
        except OSError as exc:
            self._end(PeerGone(f"could not read: {exc}"))
        else:
            self._end(PeerGone("the stream ended"))

    def _took(self, message: Mapping[str, Any]) -> None:
        """One message, put where it belongs: an answer to its id, a notification to its handler, a
        request from the other end to the method this end offers for it."""
        method, number = message.get("method"), message.get("id")
        if isinstance(method, str):
            if number is None:
                self._noticed(method, message.get("params"))
            else:
                self._requested(number, method, message.get("params"))
            return
        if number is None:
            self._end(ProtocolBroken("a message with neither a method nor an id is not one of the three shapes"))
            return
        self._answered(message)

    def _requested(self, number: Any, method: str, params: Any) -> None:
        """One request from the other end, worked out beside the reading of the stream.

        A handler is a tool call and may take as long as it likes, so it runs as a task: waiting for it
        here would stop this end from hearing the answer to its own call, which is a deadlock between two
        programs each waiting on the other. A method this end does not offer is refused at once.
        """
        if self._serving is None or not self._serving.offers(method):
            self._say(
                {
                    "jsonrpc": "2.0",
                    "id": number,
                    "error": {"code": METHOD_NOT_FOUND, "message": f"{method} is not a method this end offers"},
                }
            )
            return
        task = asyncio.create_task(self._answer_to(number, method, params))
        self._answering.add(task)
        task.add_done_callback(self._answering.discard)

    async def _answer_to(self, number: Any, method: str, params: Any) -> None:
        """Answer one request, unless the conversation ended while the method was being worked out."""
        assert self._serving is not None, "a request reaches here only when there is a table to answer it"
        answer = await self._serving.answer(number, method, params)
        if answer is not None and self._ended is None:
            self._say(answer)

    def _answered(self, message: Mapping[str, Any]) -> None:
        """One answer to the call that is waiting for it, if anybody still is."""
        number = message.get("id")
        if not isinstance(number, int):
            self._end(ProtocolBroken(f"an answer came to {number!r}, and this end numbers its requests itself"))
            return
        error = message.get("error")
        refusal: RpcError | None = None
        if error is not None:
            refusal = _refusal(error)
            if refusal is None:
                self._end(ProtocolBroken(f"the error answering {number} is not an object with a code"))
                return
        elif "result" not in message:
            self._end(ProtocolBroken(f"the answer to {number} carries neither a result nor an error"))
            return
        waiting = self._waiting.pop(number, None)
        if waiting is None:
            return
        if refusal is None:
            waiting.set_result(message.get("result"))
        else:
            waiting.set_exception(refusal)

    def _noticed(self, method: str, params: Any) -> None:
        """One notification, to whoever was declared for it — and to nobody when nobody was."""
        handler = self._notices.get(method)
        if handler is None:
            return
        try:
            handler(cast("dict[str, Any]", params) if isinstance(params, dict) else {})
        except Exception:  # noqa: BLE001 — a handler is this end's code, so its mistake is not the other end going quiet
            return

    def _end(self, why: PeerGone | ProtocolBroken) -> None:
        """Stop the conversation, and fail every caller that was waiting on an answer."""
        if self._ended is not None:
            return
        self._ended = why
        waiting, self._waiting = self._waiting, {}
        for one in waiting.values():
            if not one.done():
                one.set_exception(why)
        self._out.put_nowait(None)


def _refusal(error: Any) -> RpcError | None:
    """An error object as the exception a caller catches, or ``None`` when it is not one.

    ``None`` is what tells the reader that the other end said something that has no shape this
    protocol allows, which is a fact about the program and not about the call.
    """
    if not isinstance(error, dict):
        return None
    fields = cast("dict[str, Any]", error)
    code = fields.get("code")
    if not isinstance(code, int):
        return None
    return RpcError(code, str(fields.get("message", "")), fields.get("data"))
