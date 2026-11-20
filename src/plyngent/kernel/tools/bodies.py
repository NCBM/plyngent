"""What a tool hands the model when its answer is text: the text, or a handle when it is too long.

Every tool that produces a body asks this one question, and the *store* answers it, because how much
fits is a fact about the store and not about the tool: the threshold is
:attr:`~plyngent.kernel.documents.Documents.keep_chars`, and what comes back is either the text itself
or a document id a later call can read a window of.

It lives beside the tools, outside the document group's own module, because it is not one of them:
a listing, a file read, a fetched page and a skill body all end here, and none of them should have to
import another group's tools to say so.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plyngent.kernel.documents import TEXT_MEDIA, refuses
from plyngent.kernel.failure import Failure
from plyngent.kernel.toolresult import Document, Part, Text, ToolResult

if TYPE_CHECKING:
    from plyngent.kernel.env import Env


async def kept(env: Env, content: str, *, summary: str, media: str = TEXT_MEDIA) -> Part | ToolResult:
    """What the model is given for *content*: the text itself, or a handle when it is too long.

    ``summary`` is the line that stands in for the content in the result, so it has to say what the
    handle is — the model will read it with no body beside it.

    Content past the store's own limit comes back as a refusal, carrying the number in it: a handle
    nobody can make is not an answer, and the fix (a window) is a different call.
    """
    refusal = refuses(content)
    if refusal is not None:
        return ToolResult.of(Failure.LIMIT, refusal)
    if len(content) <= env.documents.keep_chars:
        return Text(value=content)
    handle = await env.documents.put(env.session, content, summary=summary, media=media)
    return Document(document_id=handle.id, summary=handle.summary, chars=handle.chars)


async def kept_bytes(env: Env, content: bytes, *, summary: str, media: str) -> Document | ToolResult:
    """A handle to content that is never sent as text at all — a picture, an archive, anything binary.

    Text that does not fit is still text, and a model can read a window of it; bytes are not text in
    any window, so there is no size at which handing them over would be a favour — which is why a body
    past the stored limit is refused here outright.
    """
    refusal = refuses(content)
    if refusal is not None:
        return ToolResult.of(Failure.LIMIT, refusal)
    handle = await env.documents.put(env.session, content, summary=summary, media=media)
    return Document(document_id=handle.id, summary=handle.summary, chars=handle.chars)
