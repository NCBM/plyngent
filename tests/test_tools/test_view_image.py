"""``view_image``: a local picture reaches the model as a picture."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from plyngent.agent import ToolOutput
from plyngent.tools.context import get_instance
from plyngent.tools.file.read import read_file
from plyngent.tools.file.view_image import view_image

if TYPE_CHECKING:
    from pathlib import Path

# A real 1x1 PNG: the format is sniffed from these bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


async def test_view_image_attaches_the_picture(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)

    output = await view_image.handler("shot.png")

    assert isinstance(output, ToolOutput)
    assert output.text == "image: shot.png (attached below)"
    url = output.images[0].image_url.url
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.partition(",")[2]) == PNG


async def test_the_detail_level_reaches_the_part(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)

    output = await view_image.handler("shot.png", detail="low")

    assert isinstance(output, ToolOutput)
    assert output.images[0].image_url.detail == "low"


async def test_an_unknown_detail_level_is_an_error(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)

    assert "detail must be one of" in await view_image.handler("shot.png", detail="huge")


async def test_a_non_image_file_is_an_error(workspace: Path) -> None:
    (workspace / "notes.txt").write_text("hello")

    result = await view_image.handler("notes.txt")

    assert isinstance(result, str)
    assert "not a JPEG, PNG, GIF, or WebP image" in result


async def test_a_missing_file_is_an_error(workspace: Path) -> None:
    del workspace

    assert await view_image.handler("nope.png") == "error: file not found: nope.png"


async def test_a_text_only_model_gets_the_reason_instead_of_the_picture(workspace: Path) -> None:
    """``image_in`` gates the tool exactly like an ``@path`` mention."""
    (workspace / "shot.png").write_bytes(PNG)
    instance = get_instance()
    assert instance is not None
    instance.workspace.image_in = False
    instance.workspace.image_model = ("deepseek", "deepseek-v4-pro")

    result = await view_image.handler("shot.png")

    assert isinstance(result, str)
    assert result.startswith("error: model 'deepseek-v4-pro' does not accept image input")
    assert 'set image_in = true under [providers.deepseek.models."deepseek-v4-pro"]' in result


async def test_read_file_points_at_view_image_for_a_picture(workspace: Path) -> None:
    (workspace / "shot.png").write_bytes(PNG)

    result = await read_file.handler("shot.png")

    assert result == "error: shot.png is an image (image/png); use view_image to look at it"
