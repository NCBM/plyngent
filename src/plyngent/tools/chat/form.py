from __future__ import annotations

import json
from typing import cast

from plyngent.agent import ToolTag, tool
from plyngent.prompting import FormField, NonInteractiveError, form_async
from plyngent.tools.chat.choose import parse_options


def parse_fields(raw: str) -> list[FormField]:
    """Parse form fields JSON array of objects."""
    text = raw.strip()
    if not text:
        return []
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"fields must be JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, list) or not data:
        msg = "fields must be a non-empty JSON array"
        raise ValueError(msg)
    out: list[FormField] = []
    for item_obj in cast("list[object]", data):
        if not isinstance(item_obj, dict):
            msg = "each field must be a JSON object"
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
            options = parse_options(json.dumps(options_raw))
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


@tool(name="ask_user_form", tags=ToolTag.LOCAL)
async def form_user(title: str, fields: str, *, confirm_submit: bool = True) -> str:
    """Run a multi-step form with the human; returns JSON object of answers.

    ``fields`` is a JSON array of objects:
    ``name``, ``prompt``, optional ``default``, optional ``options`` (same shape
    as ask_user_choice), optional ``allow_custom`` (default true).
    When ``confirm_submit`` is true, the human reviews a summary before submit.
    """
    try:
        parsed = parse_fields(fields)
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"
    try:
        answers = await form_async(title, parsed, confirm_submit=confirm_submit)
    except NonInteractiveError as exc:
        return f"error: {exc}"
    return json.dumps(answers, ensure_ascii=False)
