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


@dataclass(frozen=True)
class ProviderSymbolProfileMetadata:
    provider: str
    symbol: str
    status: str
    fetched_at: datetime
    sector: str | None = None
    industry: str | None = None
    asset_type: str | None = None
    quote_type: str | None = None
