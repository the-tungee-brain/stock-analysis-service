from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.broker.option_utils import portfolio_spending_by_symbol
from app.models.portfolio_news_models import (
    PortfolioHoldingsNewsItem,
    PortfolioNewsResponse,
)
from app.models.schwab_models import Position, SchwabAccounts

PORTFOLIO_NEWS_TOTAL_LIMIT = 20
PORTFOLIO_NEWS_TOP_SYMBOL_LIMIT = 6
PORTFOLIO_NEWS_PER_SYMBOL_FETCH = 8
COMPANY_NEWS_SUMMARY_MAX_LEN = 280


@dataclass(frozen=True)
class _ParsedNewsRow:
    symbol: str
    headline: str
    summary: str | None
    url: str | None
    publisher: str | None
    published_at: datetime | None
    dedupe_key: str


class PortfolioNewsService:
    def __init__(self, *, yfinance_adapter: YFinanceAdapter) -> None:
        self.yfinance_adapter = yfinance_adapter

    def build_portfolio_news(
        self,
        *,
        positions: list[Position],
        account: SchwabAccounts,
    ) -> PortfolioNewsResponse:
        liquidation = account.securitiesAccount.currentBalances.liquidationValue
        if liquidation <= 0:
            return PortfolioNewsResponse(items=[])

        spending_by_symbol = portfolio_spending_by_symbol(positions)
        ranked_symbols = sorted(
            spending_by_symbol.keys(),
            key=lambda symbol: spending_by_symbol.get(symbol, 0.0),
            reverse=True,
        )[:PORTFOLIO_NEWS_TOP_SYMBOL_LIMIT]

        rows: list[_ParsedNewsRow] = []
        for symbol in ranked_symbols:
            for raw in self.yfinance_adapter.get_news(
                symbol,
                count=PORTFOLIO_NEWS_PER_SYMBOL_FETCH,
            ):
                row = self._to_parsed_row(symbol=symbol, raw=raw)
                if row is None:
                    continue
                rows.append(row)

        merged = self._merge_and_limit(
            rows,
            weight_by_symbol=spending_by_symbol,
            liquidation=liquidation,
        )
        items = [
            PortfolioHoldingsNewsItem(
                symbol=row.symbol,
                headline=row.headline,
                source=row.publisher,
                summary=self._truncate_summary(row.summary),
                url=row.url,
                weight_pct=(
                    (spending_by_symbol.get(row.symbol, 0.0) / liquidation) * 100.0
                    if liquidation > 0
                    else None
                ),
                published_at=row.published_at,
            )
            for row in merged
        ]
        return PortfolioNewsResponse(items=items)

    @staticmethod
    def _truncate_summary(summary: str | None) -> str | None:
        if not summary:
            return None
        trimmed = summary.strip()
        if len(trimmed) <= COMPANY_NEWS_SUMMARY_MAX_LEN:
            return trimmed
        return f"{trimmed[: COMPANY_NEWS_SUMMARY_MAX_LEN - 1].rstrip()}…"

    def _to_parsed_row(
        self,
        *,
        symbol: str,
        raw: dict,
    ) -> _ParsedNewsRow | None:
        content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
        if not isinstance(content, dict):
            return None

        headline = (content.get("title") or "").strip()
        if not headline:
            return None

        summary = (
            content.get("summary")
            or content.get("description")
            or None
        )
        if isinstance(summary, str):
            summary = summary.strip() or None
        else:
            summary = None

        url = self._extract_url(content)
        provider = content.get("provider")
        publisher = None
        if isinstance(provider, dict):
            publisher = provider.get("displayName") or provider.get("name")
        if isinstance(publisher, str):
            publisher = publisher.strip() or None

        published_at = self._parse_pub_date(content.get("pubDate"))
        content_id = content.get("id")
        dedupe_key = url or f"{symbol}:{content_id or headline}"
        return _ParsedNewsRow(
            symbol=symbol.upper(),
            headline=headline,
            summary=summary,
            url=url,
            publisher=publisher,
            published_at=published_at,
            dedupe_key=dedupe_key,
        )

    @staticmethod
    def _extract_url(content: dict) -> str | None:
        canonical = content.get("canonicalUrl")
        if isinstance(canonical, dict):
            url = canonical.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        click = content.get("clickThroughUrl")
        if isinstance(click, dict):
            url = click.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        link = content.get("link")
        if isinstance(link, str) and link.strip():
            return link.strip()
        return None

    @staticmethod
    def _parse_pub_date(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        text = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    def _merge_and_limit(
        self,
        rows: list[_ParsedNewsRow],
        *,
        weight_by_symbol: dict[str, float],
        liquidation: float,
    ) -> list[_ParsedNewsRow]:
        seen: set[str] = set()
        unique: list[_ParsedNewsRow] = []
        for row in rows:
            key = row.dedupe_key.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)

        epoch = datetime.min.replace(tzinfo=None)

        def sort_key(row: _ParsedNewsRow) -> tuple:
            published = row.published_at or epoch
            if published.tzinfo is not None:
                published = published.replace(tzinfo=None)
            weight = weight_by_symbol.get(row.symbol, 0.0)
            return (-published.timestamp(), -weight)

        unique.sort(key=sort_key)
        return unique[:PORTFOLIO_NEWS_TOTAL_LIMIT]
