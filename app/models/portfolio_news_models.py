from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

_PORTFOLIO_NEWS_CONFIG = ConfigDict(populate_by_name=True)


class PortfolioHoldingsNewsItem(BaseModel):
    model_config = _PORTFOLIO_NEWS_CONFIG

    symbol: str
    headline: str
    source: str | None = None
    summary: str | None = None
    url: str | None = None
    weight_pct: float | None = Field(default=None, serialization_alias="weightPct")
    published_at: datetime | None = Field(
        default=None, serialization_alias="publishedAt"
    )


class PortfolioNewsResponse(BaseModel):
    model_config = _PORTFOLIO_NEWS_CONFIG

    items: list[PortfolioHoldingsNewsItem] = Field(default_factory=list)
