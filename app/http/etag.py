from __future__ import annotations

import hashlib
import json
from typing import Any


def json_weak_etag(payload: dict[str, Any]) -> str:
    """Stable weak ETag for JSON API payloads."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


def normalize_if_none_match(header: str | None) -> str | None:
    if not header:
        return None
    value = header.strip()
    if value.startswith("W/"):
        value = value[2:].strip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value or None
