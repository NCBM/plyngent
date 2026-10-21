"""Image parts: format sniffing, data URLs, and the inline size policy."""

from __future__ import annotations

import base64

from plyngent.agent.images import (
    MAX_INLINE_IMAGE_BYTES,
    image_part_from_bytes,
    image_part_from_url,
    sniff_image_media_type,
    split_data_url,
    to_data_url,
    user_content,
)
from plyngent.lmproto.openai_compatible.model import ChatImageUrl, ChatTextPart

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_the_format_is_sniffed_from_the_bytes() -> None:
    assert sniff_image_media_type(PNG) == "image/png"
    assert sniff_image_media_type(b"\xff\xd8\xff\xe0JFIF") == "image/jpeg"
    assert sniff_image_media_type(b"GIF89a....") == "image/gif"
    assert sniff_image_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert sniff_image_media_type(b"not an image") is None
    assert sniff_image_media_type(b"") is None


def test_a_data_url_round_trips() -> None:
    url = to_data_url("image/png", PNG)

    assert url.startswith("data:image/png;base64,")
    assert split_data_url(url) == ("image/png", base64.b64encode(PNG).decode("ascii"))
    assert split_data_url("https://example.com/a.png") is None


def test_bytes_become_an_image_part_with_a_detail_level() -> None:
    part = image_part_from_bytes(PNG, detail="low")

    assert not isinstance(part, str)
    assert part.image_url.detail == "low"
    assert part.image_url.url.startswith("data:image/png;base64,")


def test_non_image_bytes_are_an_error_text() -> None:
    result = image_part_from_bytes(b"plain text")

    assert isinstance(result, str)
    assert "not a JPEG, PNG, GIF, or WebP image" in result


def test_an_oversized_image_is_refused() -> None:
    result = image_part_from_bytes(PNG + b"\x00" * (MAX_INLINE_IMAGE_BYTES + 1))

    assert isinstance(result, str)
    assert "inline limit" in result


def test_a_url_becomes_an_image_part() -> None:
    part = image_part_from_url("https://example.com/a.jpg", detail="original")

    assert part.image_url == ChatImageUrl(url="https://example.com/a.jpg", detail="original")


def test_user_content_stays_a_string_without_images() -> None:
    assert user_content("hello") == "hello"
    assert user_content("hello", []) == "hello"


def test_user_content_becomes_a_part_array_with_images() -> None:
    part = image_part_from_url("https://example.com/a.jpg")

    content = user_content("look", [part])

    assert content == [ChatTextPart(text="look"), part]
    # An image-only turn sends no empty text part.
    assert user_content("", [part]) == [part]
