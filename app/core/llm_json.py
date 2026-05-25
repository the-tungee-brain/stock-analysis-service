import json
import re
from typing import Any, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def strip_code_fence(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _strictify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    for key in ("$defs", "definitions"):
        defs = schema.get(key)
        if isinstance(defs, dict):
            for def_schema in defs.values():
                if isinstance(def_schema, dict):
                    _strictify_schema(def_schema)

    properties = schema.get("properties")
    if isinstance(properties, dict):
        for prop in properties.values():
            if isinstance(prop, dict):
                _strictify_schema(prop)

    items = schema.get("items")
    if isinstance(items, dict):
        _strictify_schema(items)

    if schema.get("type") == "object":
        schema["additionalProperties"] = False
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            schema["required"] = list(properties.keys())
            for prop in properties.values():
                if isinstance(prop, dict):
                    prop.pop("default", None)

    return schema


def openai_response_schema(model: Type[BaseModel]) -> dict[str, Any]:
    return _strictify_schema(model.model_json_schema())


def parse_llm_json(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])

    if not isinstance(payload, dict):
        raise json.JSONDecodeError("Expected JSON object", cleaned, 0)
    return payload


def validate_llm_model(text: str, response_model: Type[T]) -> T:
    return response_model.model_validate(parse_llm_json(text))
