from typing import Literal

from pydantic import BaseModel, Field

AssetType = Literal[
    "STOCK",
    "ETF",
    "MUTUAL_FUND",
    "INDEX",
    "CRYPTO",
    "ADR",
    "BOND",
    "OPTION",
]


class TickerSymbolItem(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    title: str | None = None
    asset_type: AssetType | None = Field(default=None, serialization_alias="assetType")
    logo_url: str | None = Field(default=None, serialization_alias="logoUrl")
