"""Model-facing ``view_image``: hand a local picture to the model as a picture."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from plyngent.agent import ToolOutput, ToolTag, tool
from plyngent.agent.images import image_part_from_bytes
from plyngent.tools.workspace import AccessMode, resolve_path

if TYPE_CHECKING:
    from plyngent.lmproto.openai_compatible.model import ImageDetail

_DETAILS: frozenset[str] = frozenset({"low", "high", "original", "auto"})


@tool(tags=ToolTag.LOCAL | ToolTag.INSTANCE_STATE | ToolTag.READ_ONLY)
async def view_image(path: str, detail: str = "auto") -> str | ToolOutput:
    """Look at an image file (JPEG, PNG, GIF, or WebP) under the workspace.

    Use it for a screenshot, a chart, a diagram, or any other picture, instead
    of ``read_file`` (which would return mojibake for those bytes).

    ``detail`` trades tokens for fine visual detail: ``low`` downscales to
    512x512 before inference — fast and cheap when only the layout matters, but
    small text stops being readable — while ``high``/``original`` keep the
    picture as it is, and ``auto`` (the default) lets the provider decide.

    The format is read from the file's own bytes, so a misnamed file still
    works; anything else comes back as an error.
    """
    if detail not in _DETAILS:
        return f"error: detail must be one of {', '.join(sorted(_DETAILS))}"
    target = resolve_path(path, required=AccessMode.READ)
    if not target.exists():
        return f"error: file not found: {path}"
    if not target.is_file():
        return f"error: not a file: {path}"
    part = image_part_from_bytes(target.read_bytes(), detail=cast("ImageDetail", detail))
    if isinstance(part, str):
        return f"error: {path}: {part.removeprefix('error: ')}"
    return ToolOutput(text=f"image: {path} (attached below)", images=[part])
