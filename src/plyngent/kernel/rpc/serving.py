"""The end that is asked: the methods one side of a conversation offers, and the answer to one call.

:class:`~plyngent.kernel.rpc.peer.Peer` is the end that asks and the end that is told; this is the third
thing a channel can be, and the one a program talking back to this kernel needs. A request arriving from
the other end is a call *this* end has to carry out, and the two rules are the two that a call has:

* **The table is fixed when the conversation is set up.** A method registered halfway through is a method
  whose earlier calls were refused, which is a conversation with an order nobody can reason about.
* **Every request is answered, even the one whose method raised.** The other end is holding a future open
  on this answer, so a refusal is a result and silence is a hang.

What this does not decide is what a method *means*: a name, a handler, and an argument package is the
whole vocabulary, because the caller above this module is the one that knows whether it is talking
about a tool, a document or nothing at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from typing import Any

#: The number JSON-RPC gives a request this end took on and could not carry out: the method was there,
#: and what it does failed. The other answer a request gets is "no such method", which the peer gives
#: when it has nothing to hand it to.
INTERNAL_ERROR = -32603

#: One method: what it is handed, and what it answers with. The params are the caller's own package, so
#: a handler reads them the way a function reads its arguments, and the answer is whatever can be written
#: as one JSON value.
type Method = Callable[[Mapping[str, Any]], Awaitable[Any]]


class Serving:
    """What one end of a JSON-RPC conversation may be asked, and the answer to one request.

    Handed to a :class:`~plyngent.kernel.rpc.peer.Peer` at construction for the reason its notifications
    are: the set of methods is a fact about the protocol above it, and not a thing that grows.
    """

    _methods: Mapping[str, Method]

    def __init__(self, methods: Mapping[str, Method]) -> None:
        self._methods = dict(methods)

    @override
    def __repr__(self) -> str:
        return f"Serving({', '.join(sorted(self._methods))})"

    def offers(self, method: str) -> bool:
        """Whether this end can be asked *method* at all."""
        return method in self._methods

    async def answer(self, number: Any, method: str, params: Any) -> dict[str, Any] | None:
        """The message that answers one request, or ``None`` when this end does not offer the method.

        ``None`` is the peer's to turn into "no such method", because a table with no entry and a
        conversation with no table are the same answer to the other end.
        """
        handler = self._methods.get(method)
        if handler is None:
            return None
        try:
            given = await handler(cast("Mapping[str, Any]", params) if isinstance(params, dict) else {})
        except Exception as exc:  # noqa: BLE001 — the other end is waiting, and a refusal is not silence
            return {"jsonrpc": "2.0", "id": number, "error": {"code": INTERNAL_ERROR, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": number, "result": given}
