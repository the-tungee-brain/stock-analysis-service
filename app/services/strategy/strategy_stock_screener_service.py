from __future__ import annotations

import copy
import logging
import math
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from app.models.screener_preset_models import (
    EquityQueryClause,
    EquityQuerySpec,
    ScreenerPreset,
)
from app.models.strategy_models import (
    InvestmentStrategy,
    ScreenerResultSection,
    StrategyScreenerFilters,
    StrategyScreenerQuote,
    StrategyStockScreenerResult,
    UserInvestmentProfile,
)
from app.broker.strategy_symbol_alignment import strategy_symbol_list
from app.screener.equity_query_compiler import compile_equity_query
from app.screener.etf_universe_screener import screen_etf_preset
from app.screener.preset_registry import (
    STRATEGY_COMPANION_PRESET_IDS,
    get_preset,
    preset_for_strategy,
    preset_summary,
)

logger = logging.getLogger(__name__)

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)

DEFAULT_SCREEN_SIZE = 100
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 50
MAX_SCREEN_SIZE = 250
YF_SCREEN_SORT: dict[str, Any] = {
    "sortField": "intradaymarketcap",
    "sortAsc": False,
}

# API override keys -> preset equity clause field names.
OVERRIDE_FIELD_MAP = {
    "min_market_cap": ("marketCap", "gte"),
    "max_pe": ("trailingPE", "lte"),
    "min_dividend_yield": ("dividendYield", "gte"),
    "max_dividend_yield": ("dividendYield", "lte"),
}


class StrategyStockScreenerService:
    @staticmethod
    def supports_stock_screener(strategy: InvestmentStrategy) -> bool:
        return strategy in {
            InvestmentStrategy.WHEEL,
            InvestmentStrategy.CSP_INCOME,
            InvestmentStrategy.COVERED_CALL,
            InvestmentStrategy.DIVIDEND,
            InvestmentStrategy.ETF_CORE,
        }

    @staticmethod
    def resolve_preset(
        strategy: InvestmentStrategy,
        profile: UserInvestmentProfile,
        *,
        preset_id: str | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> ScreenerPreset:
        preset = copy.deepcopy(
            preset_for_strategy(strategy)
            if preset_id is None
            else _load_preset_or_raise(preset_id)
        )
        preset = _apply_profile_adjustments(preset, profile, strategy)
        if overrides:
            preset = _apply_api_overrides(preset, overrides)
        return preset

    @staticmethod
    def describe_preset(preset: ScreenerPreset) -> str:
        if preset.equity_query is None:
            structure = preset.post_filters.get("structure") or {}
            examples = structure.get("examples_preferred") or []
            if examples:
                return f"{preset.label}: {', '.join(str(item) for item in examples[:5])}"
            return preset.description

        parts: list[str] = []
        for clause in preset.equity_query.clauses:
            if clause.field == "marketCap" and clause.op == "gte" and clause.value is not None:
                parts.append(f"≥ ${float(clause.value) / 1e9:.0f}B market cap")
            elif (
                clause.field == "regularMarketPrice"
                and clause.op == "gte"
                and clause.value is not None
            ):
                parts.append(f"price ≥ ${clause.value:.0f}")
            elif (
                clause.field == "regularMarketPrice"
                and clause.op == "lte"
                and clause.value is not None
            ):
                parts.append(f"price ≤ ${clause.value:.0f}")
            elif clause.field == "trailingPE" and clause.op == "lte" and clause.value is not None:
                parts.append(f"P/E ≤ {float(clause.value):.0f}")
            elif (
                clause.field == "dividendYield"
                and clause.op == "gte"
                and clause.value is not None
            ):
                parts.append(f"yield ≥ {float(clause.value) * 100:.1f}%")
            elif clause.field == "sector" and clause.op == "is-in" and clause.values:
                parts.append(f"{len(clause.values)} sectors")
        return " · ".join(parts) if parts else preset.description

    @staticmethod
    def _map_equity_quote(raw: dict) -> StrategyScreenerQuote:
        market_cap = raw.get("marketCap")
        if market_cap is None:
            market_cap = raw.get("intradaymarketcap")

        dividend_yield = raw.get("dividendYield")
        if dividend_yield is None:
            dividend_yield = raw.get("forward_dividend_yield")

        pe_ratio = raw.get("trailingPE")
        if pe_ratio is None:
            pe_ratio = raw.get("peratio.lasttwelvemonths")

        price = raw.get("regularMarketPrice")
        if price is None:
            price = raw.get("intradayprice")

        return StrategyScreenerQuote(
            symbol=str(raw.get("symbol", "")).upper(),
            company_name=raw.get("shortName") or raw.get("longName") or raw.get("displayName"),
            sector=raw.get("sector"),
            market_cap=float(market_cap) if market_cap is not None else None,
            pe_ratio=float(pe_ratio) if pe_ratio is not None else None,
            dividend_yield=float(dividend_yield) if dividend_yield is not None else None,
            price=float(price) if price is not None else None,
        )

    def screen_stocks(
        self,
        *,
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        preset_id: str | None = None,
        overrides: dict[str, Any] | None = None,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> StrategyStockScreenerResult | None:
        if not self.supports_stock_screener(strategy):
            return None

        preset = self.resolve_preset(
            strategy,
            profile,
            preset_id=preset_id,
            overrides=overrides,
        )
        resolved_page = max(1, page)
        resolved_page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        offset = (resolved_page - 1) * resolved_page_size

        try:
            if preset.equity_query is not None:
                query = compile_equity_query(preset.equity_query)
                raw = yf.screen(
                    query,
                    size=resolved_page_size,
                    offset=offset,
                    **YF_SCREEN_SORT,
                )
                quotes_raw = raw.get("quotes") or []
                total = int(raw.get("total") or len(quotes_raw))
                quotes = [
                    quote
                    for item in quotes_raw
                    if (quote := self._map_equity_quote(item)).symbol
                ]
            else:
                quotes, total = screen_etf_preset(
                    preset,
                    limit=resolved_page_size,
                )
                resolved_page = 1
                offset = 0
        except Exception:
            logger.exception(
                "Screener failed preset=%s strategy=%s user=%s",
                preset.id,
                strategy.value,
                profile.user_id,
            )
            return None

        total_pages = max(1, math.ceil(total / resolved_page_size)) if total else 1
        summary = self.describe_preset(preset)
        sections = self._companion_sections(
            strategy=strategy,
            preset_id=preset_id,
        )
        screener_filters = filters_from_preset(preset)
        pinned_quotes = self._fetch_pinned_quotes(
            profile=profile,
            strategy=strategy,
            filters=screener_filters,
        )

        return StrategyStockScreenerResult(
            strategy=strategy,
            preset=preset_summary(preset),
            filters=screener_filters,
            quotes=quotes,
            pinned_quotes=pinned_quotes,
            total_count=total,
            page=resolved_page,
            page_size=resolved_page_size,
            total_pages=total_pages,
            summary=summary,
            sections=sections,
            generated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _companion_sections(
        *,
        strategy: InvestmentStrategy,
        preset_id: str | None,
    ) -> list[ScreenerResultSection]:
        if preset_id is not None:
            return []

        sections: list[ScreenerResultSection] = []
        for companion_id in STRATEGY_COMPANION_PRESET_IDS.get(strategy, []):
            companion = get_preset(companion_id)
            if companion is None or companion.equity_query is not None:
                continue
            quotes, total = screen_etf_preset(
                companion,
                limit=20,
            )
            sections.append(
                ScreenerResultSection(
                    preset=preset_summary(companion),
                    quotes=quotes,
                    total_count=total,
                    page=1,
                    page_size=max(len(quotes), 1),
                    total_pages=1,
                )
            )
        return sections

    @staticmethod
    def _fetch_pinned_quotes(
        *,
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        filters: StrategyScreenerFilters | None,
    ) -> list[StrategyScreenerQuote]:
        symbols = strategy_symbol_list(profile)
        if not symbols:
            return []

        pinned: list[StrategyScreenerQuote] = []
        for symbol in symbols:
            upper = symbol.upper()
            if not upper:
                continue
            try:
                info = yf.Ticker(upper).info or {}
            except Exception:
                logger.debug("Unable to load pinned quote for %s", upper)
                continue

            quote = StrategyStockScreenerService._map_equity_quote(
                {
                    "symbol": upper,
                    "shortName": info.get("shortName") or info.get("longName"),
                    "sector": info.get("sector") or info.get("category"),
                    "marketCap": info.get("marketCap") or info.get("totalAssets"),
                    "trailingPE": info.get("trailingPE"),
                    "dividendYield": info.get("dividendYield") or info.get("yield"),
                    "regularMarketPrice": info.get("regularMarketPrice"),
                }
            )
            if not quote.symbol:
                continue
            quote = quote.model_copy(
                update={
                    "preset_fit": _quote_meets_filters(quote, filters),
                }
            )
            pinned.append(quote)
        return pinned


def _quote_meets_filters(
    quote: StrategyScreenerQuote,
    filters: StrategyScreenerFilters | None,
) -> bool | None:
    if filters is None:
        return None
    if filters.min_market_cap and quote.market_cap is not None:
        if quote.market_cap < filters.min_market_cap:
            return False
    if filters.max_pe is not None and quote.pe_ratio is not None:
        if quote.pe_ratio > filters.max_pe:
            return False
    if filters.require_dividend:
        if quote.dividend_yield is None or quote.dividend_yield <= 0:
            return False
    if filters.min_dividend_yield is not None and quote.dividend_yield is not None:
        if quote.dividend_yield < filters.min_dividend_yield:
            return False
    return True


def _load_preset_or_raise(preset_id: str) -> ScreenerPreset:
    from app.screener.preset_registry import get_preset

    preset = get_preset(preset_id)
    if preset is None:
        raise KeyError(f"Unknown screener preset: {preset_id}")
    return copy.deepcopy(preset)


def _apply_profile_adjustments(
    preset: ScreenerPreset,
    profile: UserInvestmentProfile,
    strategy: InvestmentStrategy,
) -> ScreenerPreset:
    if preset.equity_query is None:
        return preset

    risk = profile.risk_tolerance or "moderate"
    overrides: dict[str, Any] = {}

    if strategy in WHEEL_LIKE:
        if risk == "conservative":
            overrides = {"min_market_cap": 10_000_000_000, "max_pe": 35.0}
        elif risk == "aggressive":
            overrides = {"min_market_cap": 2_000_000_000, "max_pe": 60.0}

    if strategy == InvestmentStrategy.DIVIDEND:
        if profile.dividend and profile.dividend.target_yield_pct is not None:
            overrides["min_dividend_yield"] = max(
                0.0,
                profile.dividend.target_yield_pct / 100.0,
            )
        if risk == "aggressive":
            overrides.setdefault("min_market_cap", 1_000_000_000)
        elif risk == "conservative":
            overrides.setdefault("min_market_cap", 5_000_000_000)
            overrides.setdefault("max_pe", 22.0)

    if not overrides:
        return preset

    return _apply_api_overrides(preset, overrides)


def _apply_api_overrides(
    preset: ScreenerPreset,
    overrides: dict[str, Any],
) -> ScreenerPreset:
    if preset.equity_query is None:
        return preset

    spec = preset.equity_query.model_copy(deep=True)

    if "sectors" in overrides:
        sector_values = overrides["sectors"]
        if sector_values:
            spec = _upsert_clause(
                spec,
                EquityQueryClause(
                    op="is-in",
                    field="sector",
                    values=list(sector_values),
                ),
            )
        else:
            spec = _remove_clauses(spec, {"sector"})

    require_dividend = overrides.get("require_dividend")
    if require_dividend is True:
        spec = _upsert_clause(
            spec,
            EquityQueryClause(
                op="gt",
                field="dividendYield",
                value=0.0001,
            ),
        )
    elif require_dividend is False:
        spec = _remove_clauses(spec, {"dividendYield"})

    for override_key, (field, op) in OVERRIDE_FIELD_MAP.items():
        if override_key not in overrides or overrides[override_key] is None:
            continue
        spec = _upsert_clause(
            spec,
            EquityQueryClause(
                op=op,  # type: ignore[arg-type]
                field=field,
                value=overrides[override_key],
            ),
        )

    return preset.model_copy(update={"equity_query": spec})


def _upsert_clause(spec: EquityQuerySpec, clause: EquityQueryClause) -> EquityQuerySpec:
    kept = [existing for existing in spec.clauses if existing.field != clause.field]
    kept.append(clause)
    return spec.model_copy(update={"clauses": kept})


def _remove_clauses(spec: EquityQuerySpec, fields: set[str]) -> EquityQuerySpec:
    kept = [clause for clause in spec.clauses if clause.field not in fields]
    return spec.model_copy(update={"clauses": kept})


def filters_from_preset(preset: ScreenerPreset) -> StrategyScreenerFilters | None:
    if preset.equity_query is None:
        return None

    min_market_cap = 5_000_000_000
    max_pe: float | None = None
    require_dividend = False
    min_dividend_yield: float | None = None
    sectors: list[str] | None = None

    for clause in preset.equity_query.clauses:
        if clause.field == "marketCap" and clause.op == "gte" and clause.value is not None:
            min_market_cap = int(clause.value)
        elif clause.field == "trailingPE" and clause.op == "lte" and clause.value is not None:
            max_pe = float(clause.value)
        elif clause.field == "dividendYield":
            if clause.op in {"gt", "gte"} and clause.value is not None:
                require_dividend = True
                min_dividend_yield = float(clause.value)
            elif clause.op == "lte" and clause.value is not None:
                pass
        elif clause.field == "sector" and clause.op == "is-in" and clause.values:
            sectors = [str(value) for value in clause.values]

    filters = StrategyScreenerFilters(
        min_market_cap=min_market_cap,
        max_pe=max_pe,
        require_dividend=require_dividend,
        min_dividend_yield=min_dividend_yield,
        sectors=sectors,
        exchanges=["NMS", "NYQ"],
    )
    return filters
