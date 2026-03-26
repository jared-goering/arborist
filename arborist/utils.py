"""Shared utilities — JSON parsing helpers."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(value: str | dict | None) -> dict[str, Any] | None:
    """Parse a JSON string to a dict, or return the dict as-is.

    Returns None if *value* is None or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_json_field(value: str | dict | list | None) -> Any:
    """Parse a JSON string that may be a dict *or* list.

    Returns the original *value* unchanged when parsing fails (lenient).
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


def parse_json_robust(text: str) -> Any:
    """Parse JSON from LLM output, handling markdown fences, comments, etc."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip single-line comments (// ...)
    cleaned = re.sub(r"//[^\n]*", "", text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON array from the text
    array_match = re.search(r"\[[\s\S]*\]", text)
    if array_match:
        try:
            return json.loads(array_match.group())
        except json.JSONDecodeError:
            pass

    # Try to extract a JSON object with a "configs" key
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "configs" in parsed:
                return parsed["configs"]
            return parsed
        except json.JSONDecodeError:
            pass

    return None
