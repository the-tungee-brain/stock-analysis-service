from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

from app.models.screener_preset_models import (
    EquityQueryClause,
    EquityQuerySpec,
    ScreenerPreset,
    ScreenerPresetSummary,
)
from app.models.screener_preset_models import ScreenerPresetSummary
from app.models.strategy_models import (
    InvestmentStrategy,
    StrategyScreenerQuote,
    StrategyStockScreenerResult,
    UserInvestmentProfile,
)
from app.screener.equity_query_compiler import compile_equity_query
from app.screener.etf_universe_screener import screen_etf_preset
from app.screener.preset_registry import preset_for_strategy, preset_summary

logger = logging.getLogger(__name__)

WHEEL_LIKE = frozenset(
    {
        InvestmentStrategy.WHEEL,
        InvestmentStrategy.CSP_INCOME,
        InvestmentStrategy.COVERED_CALL,
    }
)

DEFAULT_SCREEN_SIZE = 100
MAX_SCREEN_SIZE = 250

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
    def _profile_symbols_for_strategy(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
    ) -> list[str]:
        if strategy in WHEEL_LIKE and profile.wheel:
            return list(profile.wheel.wheel_symbols or [])
        if strategy == InvestmentStrategy.DIVIDEND and profile.dividend:
            return list(profile.dividend.dividend_symbols or [])
        if strategy == InvestmentStrategy.ETF_CORE and profile.etf_core:
            return list((profile.etf_core.target_allocation or {}).keys())
        return []

    @staticmethod
    def existing_symbols(
        profile: UserInvestmentProfile,
        strategy: InvestmentStrategy,
        *,
        held_symbols: list[str] | None = None,
    ) -> set[str]:
        symbols = {
            symbol.upper()
            for symbol in StrategyStockScreenerService._profile_symbols_for_strategy(
                profile, strategy
            )
            if symbol
        }
        for symbol in held_symbols or []:
            if symbol:
                symbols.add(symbol.upper())
        return symbols

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
        limit: int = DEFAULT_SCREEN_SIZE,
        held_symbols: list[str] | None = None,
    ) -> StrategyStockScreenerResult | None:
        if not self.supports_stock_screener(strategy):
            return None

        preset = self.resolve_preset(
            strategy,
            profile,
            preset_id=preset_id,
            overrides=overrides,
        )
        resolved_limit = max(1, min(limit, MAX_SCREEN_SIZE))
        exclude = self.existing_symbols(profile, strategy, held_symbols=held_symbols)

        try:
            if preset.equity_query is not None:
                query = compile_equity_query(preset.equity_query)
                raw = yf.screen(
                    query,
                    size=resolved_limit,
                    sortField="intradaymarketcap",
                    sortAsc=False,
                )
                quotes_raw = raw.get("quotes") or []
                total = int(raw.get("total") or len(quotes_raw))
                quotes = [
                    quote
                    for item in quotes_raw
                    if (quote := self._map_equity_quote(item)).symbol
                    and quote.symbol not in exclude
                ]
            else:
                quotes, total = screen_etf_preset(
                    preset,
                    exclude=exclude,
                    limit=resolved_limit,
                )
        except Exception:
            logger.exception(
                "Screener failed preset=%s strategy=%s user=%s",
                preset.id,
                strategy.value,
                profile.user_id,
            )
            return None

        summary = self.describe_preset(preset)

        return StrategyStockScreenerResult(
            strategy=strategy,
            preset=preset_summary(preset),
            quotes=quotes,
            total_count=total,
            summary=summary,
            generated_at=datetime.now(timezone.utc),
        )


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

    if "sectors" in overrides and overrides["sectors"] is not None:
        spec = _upsert_clause(
            spec,
            EquityQueryClause(
                op="is-in",
                field="sector",
                values=list(overrides["sectors"]),
            ),
        )

    if overrides.get("require_dividend") is False:
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
