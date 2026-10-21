"""Image parts for the model: data URLs, format sniffing, and the size policy.

A model-visible image is a :class:`ChatImagePart` holding a URL — an ``http(s)``
link or an inline ``data:`` URL. Providers sniff the real format from the bytes
(never from a file name), so this module sniffs too and builds the data URL from
what it found.

Inline bytes are capped at :data:`MAX_INLINE_IMAGE_BYTES`, well under the 32 MiB
a provider accepts: every stored message is re-sent on each round and lives in
the session's JSON message column, so a huge screenshot would be paid for over
and over. A larger image has to be resized first.
"""

from __future__ import annotations

import base64

from plyngent.lmproto.openai_compatible.model import ChatImagePart, ChatImageUrl, ImageDetail

# Inline payload ceiling for one image (the API allows 32 MiB; see the module docstring).
MAX_INLINE_IMAGE_BYTES = 8 * 1024 * 1024

# Formatted images the API accepts, by their leading bytes.
_IMAGE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)
# RIFF....WEBP: four bytes of container, four of size, then the form type.
_WEBP_HEADER_LEN = 12


def sniff_image_media_type(data: bytes) -> str | None:
    """Media type of an image *data* by its content, or ``None`` if unsupported."""
    for signature, media_type in _IMAGE_SIGNATURES:
        if data.startswith(signature):
            return media_type
    # WebP: RIFF....WEBP
    if len(data) >= _WEBP_HEADER_LEN and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def to_data_url(media_type: str, data: bytes) -> str:
    """Encode *data* as an inline ``data:`` URL of *media_type*."""
    return f"data:{media_type};base64,{base64.b64encode(data).decode('ascii')}"


def split_data_url(url: str) -> tuple[str, str] | None:
    """``(media_type, base64_payload)`` of a ``data:`` URL, or ``None`` otherwise."""
    if not url.startswith("data:"):
        return None
    header, _, payload = url.partition(",")
    media_type = header[len("data:") :].split(";")[0].strip()
    if not media_type or not payload:
        return None
    return media_type, payload


def image_part_from_bytes(data: bytes, *, detail: ImageDetail = "auto") -> ChatImagePart | str:
    """An image part for *data*; a string is the model-facing error instead."""
    media_type = sniff_image_media_type(data)
    if media_type is None:
        return "error: not a JPEG, PNG, GIF, or WebP image"
    if len(data) > MAX_INLINE_IMAGE_BYTES:
        return (
            f"error: image is {len(data)} bytes, over the {MAX_INLINE_IMAGE_BYTES}-byte inline limit; "
            "resize it and try again"
        )
    return ChatImagePart(image_url=ChatImageUrl(url=to_data_url(media_type, data), detail=detail))


def image_part_from_url(url: str, *, detail: ImageDetail = "auto") -> ChatImagePart:
    """An image part for a public ``http(s)`` URL or an inline ``data:`` URL."""
    return ChatImagePart(image_url=ChatImageUrl(url=url, detail=detail))
