from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ProviderSymbolProfile:
    provider: str
    symbol: str
    fetched_at: datetime
    raw_json: dict[str, Any]
