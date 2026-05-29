from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

from app.models.finnhub_news_models import NewsItem

YFinanceNewsTab = Literal["news", "all", "press releases"]


@dataclass(frozen=True)
class ParsedYFinanceNews:
    headline: str
    summary: str | None
    url: str
    source: str
    published_at: datetime | None
    content_type: str | None


def parse_yfinance_news_item(raw: dict[str, Any]) -> ParsedYFinanceNews | None:
    content = raw.get("content") if isinstance(raw.get("content"), dict) else raw
    if not isinstance(content, dict):
        return None

    headline = (content.get("title") or "").strip()
    if not headline:
        return None

    url = _extract_url(content)
    if not url:
        return None

    summary = content.get("summary") or content.get("description")
    if isinstance(summary, str):
        summary = summary.strip() or None
    else:
        summary = None

    provider = content.get("provider")
    source = ""
    if isinstance(provider, dict):
        source = (provider.get("displayName") or provider.get("name") or "").strip()

    published_at = _parse_pub_date(content.get("pubDate"))
    content_type = content.get("contentType")
    if isinstance(content_type, str):
        content_type = content_type.strip() or None
    else:
        content_type = None

    return ParsedYFinanceNews(
        headline=headline,
        summary=summary,
        url=url,
        source=source or "Press release",
        published_at=published_at,
        content_type=content_type,
    )


def parsed_to_news_item(
    *,
    symbol: str,
    parsed: ParsedYFinanceNews,
    index: int,
    category: str = "company news",
) -> NewsItem:
    published = parsed.published_at or datetime.now(timezone.utc)
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    stable_id = int(
        hashlib.sha256(
            f"{symbol}:{parsed.url}:{parsed.headline}".encode()
        ).hexdigest()[:8],
        16,
    )

    return NewsItem(
        category=category,
        datetime=published,
        headline=parsed.headline,
        id=stable_id if stable_id else index + 1,
        image=None,
        related=symbol.upper(),
        source=parsed.source,
        summary=parsed.summary or "",
        url=parsed.url,
    )


def yfinance_raw_to_news_items(
    *,
    symbol: str,
    raw_items: list[dict[str, Any]],
    lookback_days: int | None = None,
    category: str = "company news",
) -> list[NewsItem]:
    symbol_upper = symbol.strip().upper()
    cutoff: datetime | None = None
    if lookback_days is not None and lookback_days > 0:
        cutoff = datetime.combine(
            date.today() - timedelta(days=lookback_days),
            datetime.min.time(),
            tzinfo=timezone.utc,
        )

    items: list[NewsItem] = []
    for index, raw in enumerate(raw_items):
        parsed = parse_yfinance_news_item(raw)
        if parsed is None:
            continue
        if cutoff is not None:
            published = parsed.published_at
            if published is None:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published < cutoff:
                continue
        items.append(
            parsed_to_news_item(
                symbol=symbol_upper,
                parsed=parsed,
                index=index,
                category=category,
            )
        )
    return items


def _extract_url(content: dict[str, Any]) -> str | None:
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


def _parse_pub_date(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None
