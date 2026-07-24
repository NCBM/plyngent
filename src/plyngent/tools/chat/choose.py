from __future__ import annotations

import json
from typing import cast

from plyngent.agent import ToolTag, tool
from plyngent.prompting import ChoiceOption, NonInteractiveError, choose_async


def parse_options(raw: str) -> list[ChoiceOption]:
    """Parse options JSON: list of strings or objects with label/description/value."""
    text = raw.strip()
    if not text:
        return []
    try:
        data: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"options must be JSON: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, list):
        msg = "options must be a JSON array"
        raise TypeError(msg)
    out: list[ChoiceOption] = []
    for item_obj in cast("list[object]", data):
        if isinstance(item_obj, str):
            out.append(ChoiceOption(label=item_obj))
            continue
        if isinstance(item_obj, dict):
            raw_map = {str(key): value for key, value in cast("dict[object, object]", item_obj).items()}
            label_obj = raw_map.get("label")
            if not isinstance(label_obj, str) or not label_obj:
                msg = "each option object needs a non-empty string label"
                raise ValueError(msg)
            description_obj = raw_map.get("description", "")
            value_obj = raw_map.get("value")
            out.append(
                ChoiceOption(
                    label=label_obj,
                    description=description_obj if isinstance(description_obj, str) else "",
                    value=value_obj if isinstance(value_obj, str) else None,
                )
            )
            continue
        msg = "options items must be strings or objects"
        raise TypeError(msg)
    return out


@tool(name="ask_user_choice", tags=ToolTag.LOCAL)
async def choose_user(
    question: str,
    options: str,
    default: str = "",
    *,
    allow_custom: bool = True,
) -> str:
    """Ask the human to pick from a list of options (or type a custom answer).

    ``options`` is a JSON array of strings, or objects with
    ``label``, optional ``description``, optional ``value``.
    When ``allow_custom`` is true (default), free-text answers are accepted.
    Returns the chosen option value (or custom text).
    """
    try:
        parsed = parse_options(options)
    except (TypeError, ValueError) as exc:
        return f"error: {exc}"
    if not parsed and not allow_custom:
        return "error: options must be a non-empty JSON array when allow_custom is false"
    try:
        return await choose_async(
            question,
            parsed,
            default=default or None,
            allow_custom=allow_custom,
        )
    except NonInteractiveError as exc:
        return f"error: {exc}"
