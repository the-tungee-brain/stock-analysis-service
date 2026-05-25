from pydantic import BaseModel, Field


class TickerSymbolItem(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=16)
    name: str | None = None
