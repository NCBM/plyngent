from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

from plyngent.agent import ToolTag, tool
from plyngent.prompting import FormField, NonInteractiveError, form_async
from plyngent.tools.chat.choose import parse_options
from plyngent.tools.chat.shape import first_error, list_error, string_error

if TYPE_CHECKING:
    from collections.abc import Sequence


def parse_fields(fields: Sequence[object]) -> list[FormField]:
    """Normalize a list of field dicts into ``FormField`` list."""
    if not fields:
        msg = "fields must be a non-empty list"
        raise ValueError(msg)
    out: list[FormField] = []
    for item_obj in fields:
        if not isinstance(item_obj, dict):
            msg = "each field must be an object with name/prompt"
            raise TypeError(msg)
        raw_map = {str(key): value for key, value in cast("dict[object, object]", item_obj).items()}
        name = raw_map.get("name")
        prompt = raw_map.get("prompt")
        if not isinstance(name, str) or not name:
            msg = "each field needs a non-empty string name"
            raise ValueError(msg)
        if not isinstance(prompt, str) or not prompt:
            msg = "each field needs a non-empty string prompt"
            raise ValueError(msg)
        default = raw_map.get("default")
        options_raw = raw_map.get("options")
        options = None
        if options_raw is not None:
            if not isinstance(options_raw, list):
                msg = f"field {name!r} options must be a list"
                raise TypeError(msg)
            options = parse_options(cast("list[object]", options_raw))
        allow_custom_obj = raw_map.get("allow_custom", True)
        allow_custom = allow_custom_obj if isinstance(allow_custom_obj, bool) else True
        out.append(
            FormField(
                name=name,
                prompt=prompt,
                default=default if isinstance(default, str) else None,
                options=options,
                allow_custom=allow_custom,
            )
        )
    return out


@tool(name="ask_user_form", tags=ToolTag.LOCAL | ToolTag.READ_ONLY)
async def form_user(
    title: str,
    fields: list[dict[str, object]],
    *,
    confirm_submit: bool = True,
) -> str:
    """Run a multi-step form with the human; returns JSON object of answers.

    ``fields`` is a list of objects:
    ``name`` (required), ``prompt`` (required), optional ``default``,
    optional ``options`` (same shape as ask_user_choice),
    optional ``allow_custom`` (default true).
    When ``confirm_submit`` is true, the human reviews a summary before submit.
    """
    shape_error = first_error(
        string_error("ask_user_form", "title", title),
        list_error(
            "ask_user_form",
            "fields",
            fields,
            items="objects",
            example='[{"name": "port", "prompt": "Which port?"}]',
        ),
    )
    if shape_error is not None:
        return f"error: {shape_error}"
    try:
        parsed = parse_fields(fields)
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"
    try:
        answers = await form_async(title, parsed, confirm_submit=confirm_submit)
    except NonInteractiveError as exc:
        return f"error: {exc}"
    return json.dumps(answers, ensure_ascii=False)
