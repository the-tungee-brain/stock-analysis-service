from __future__ import annotations

import logging
from typing import Any

import yfinance as yf

from app.models.screener_preset_models import ScreenerPreset
from app.models.strategy_models import StrategyScreenerQuote

logger = logging.getLogger(__name__)


def _structure_filters(preset: ScreenerPreset) -> dict[str, Any]:
    return preset.post_filters.get("structure") or {}


def _quote_from_info(symbol: str, info: dict[str, Any]) -> StrategyScreenerQuote:
    dividend_yield = info.get("dividendYield") or info.get("yield")
    return StrategyScreenerQuote(
        symbol=symbol.upper(),
        company_name=info.get("shortName") or info.get("longName"),
        sector=info.get("category") or info.get("quoteType"),
        market_cap=float(info["totalAssets"])
        if info.get("totalAssets") is not None
        else None,
        pe_ratio=None,
        dividend_yield=float(dividend_yield) if dividend_yield is not None else None,
        price=float(info["regularMarketPrice"])
        if info.get("regularMarketPrice") is not None
        else None,
    )


def _passes_etf_filters(
    info: dict[str, Any],
    *,
    structure: dict[str, Any],
    liquidity: dict[str, Any],
    price_cfg: dict[str, Any],
    dividend_cfg: dict[str, Any],
) -> bool:
    min_assets = structure.get("min_total_assets") or liquidity.get("min_total_assets")
    max_expense = structure.get("max_expense_ratio")

    total_assets = info.get("totalAssets")
    if min_assets is not None:
        if total_assets is None or float(total_assets) < float(min_assets):
            return False

    expense = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
    if max_expense is not None and expense is not None:
        if float(expense) > float(max_expense):
            return False

    regular_price = info.get("regularMarketPrice")
    if regular_price is not None:
        price_value = float(regular_price)
        min_price = price_cfg.get("min_price")
        max_price = price_cfg.get("max_price")
        if min_price is not None and price_value < float(min_price):
            return False
        if max_price is not None and price_value > float(max_price):
            return False

    raw_yield = info.get("dividendYield") or info.get("yield")
    if raw_yield is not None:
        yield_value = float(raw_yield)
        min_yield = dividend_cfg.get("min_dividend_yield")
        max_yield = dividend_cfg.get("max_dividend_yield")
        if min_yield is not None and yield_value < float(min_yield):
            return False
        if max_yield is not None and yield_value > float(max_yield):
            return False

    return True


def screen_etf_preset(
    preset: ScreenerPreset,
    *,
    limit: int,
) -> tuple[list[StrategyScreenerQuote], int]:
    structure = _structure_filters(preset)
    liquidity = preset.post_filters.get("liquidity") or {}
    price_cfg = preset.post_filters.get("price") or {}
    dividend_cfg = preset.post_filters.get("dividend") or {}
    symbols = structure.get("examples_preferred") or []

    quotes: list[StrategyScreenerQuote] = []
    for symbol in symbols:
        upper = str(symbol).upper()
        if not upper:
            continue
        try:
            info = yf.Ticker(upper).info or {}
        except Exception:
            logger.debug("Unable to load ETF info for %s", upper)
            continue

        if not _passes_etf_filters(
            info,
            structure=structure,
            liquidity=liquidity,
            price_cfg=price_cfg,
            dividend_cfg=dividend_cfg,
        ):
            continue

        quotes.append(_quote_from_info(upper, info))
        if len(quotes) >= limit:
            break

    return quotes, len(symbols)
