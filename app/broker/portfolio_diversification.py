from __future__ import annotations

from typing import TypedDict

from app.broker.option_utils import total_csp_reserved_cash
from app.broker.position_metrics import portfolio_liquidation_value
from app.broker.sector_labels import normalize_sector_label
from app.models.intelligence_models import SectorWeight
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile


class EtfFundMetrics(TypedDict):
    dividend_yield: str | None
    expense_ratio: str | None


def _market_value_by_symbol(
    ranked: list[tuple[str, float, float]],
) -> dict[str, float]:
    return {symbol: market_value for symbol, market_value, _ in ranked}


def _aggregate_symbol_weights(
    positions: list[Position],
    liquidation: float,
) -> list[tuple[str, float, float]]:
    if liquidation <= 0:
        return []

    by_symbol: dict[str, float] = {}
    for position in positions:
        instrument = position.instrument
        if instrument.assetType == "OPTION":
            symbol = (instrument.underlyingSymbol or instrument.symbol or "").upper()
        else:
            symbol = (instrument.symbol or "").upper()
        if not symbol:
            continue
        by_symbol[symbol] = by_symbol.get(symbol, 0.0) + abs(position.marketValue)

    ranked = sorted(by_symbol.items(), key=lambda item: item[1], reverse=True)
    return [
        (symbol, market_value, (market_value / liquidation) * 100.0)
        for symbol, market_value in ranked
    ]


def _short_put_underlyings(positions: list[Position]) -> list[str]:
    symbols: set[str] = set()
    for position in positions:
        instrument = position.instrument
        if instrument.assetType != "OPTION":
            continue
        if instrument.putCall != "PUT" or position.shortQuantity <= 0:
            continue
        underlying = (instrument.underlyingSymbol or instrument.symbol or "").upper()
        if underlying:
            symbols.add(underlying)
    return sorted(symbols)


def _concentration_flags(weight_pct: float, limit_pct: float) -> str:
    if weight_pct >= 30:
        return "CRITICAL (>30%)"
    if weight_pct >= 20:
        return "HIGH (20–30%)"
    if weight_pct >= 15:
        return "ELEVATED (15–20%)"
    if weight_pct >= limit_pct:
        return f"ABOVE TARGET (>{limit_pct:.0f}%)"
    return ""


def _profile_single_name_limit(profile: UserInvestmentProfile | None) -> float:
    if profile and profile.wheel and profile.wheel.max_single_name_pct:
        return float(profile.wheel.max_single_name_pct)
    if profile and profile.risk_tolerance == "conservative":
        return 12.0
    if profile and profile.risk_tolerance == "aggressive":
        return 20.0
    return 15.0


def format_diversification_summary_block(
    *,
    positions: list[Position],
    account: SchwabAccounts,
    sector_weights: list[SectorWeight] | None = None,
    profile: UserInvestmentProfile | None = None,
    etf_fund_metrics: dict[str, EtfFundMetrics] | None = None,
) -> str | None:
    if not positions:
        return None

    liquidation = portfolio_liquidation_value(account=account, positions=positions)
    if not liquidation or liquidation <= 0:
        return None

    balances = account.securitiesAccount.currentBalances
    cash = balances.cashBalance or 0.0
    csp_reserved = total_csp_reserved_cash(positions)
    cash_after_csp = max(cash - csp_reserved, 0.0)
    min_cash_buffer_pct = 5.0
    min_cash_buffer = liquidation * (min_cash_buffer_pct / 100.0)
    deployable_cash = max(cash_after_csp - min_cash_buffer, 0.0)

    ranked = _aggregate_symbol_weights(positions, liquidation)
    if not ranked:
        return None

    single_name_limit = _profile_single_name_limit(profile)
    top1 = ranked[0][2] if ranked else 0.0
    top3 = sum(weight for _, _, weight in ranked[:3])
    top5 = sum(weight for _, _, weight in ranked[:5])
    hhi = sum((weight / 100.0) ** 2 for _, _, weight in ranked)
    effective_names = (1.0 / hhi) if hhi > 0 else len(ranked)

    lines = [
        "## Portfolio concentration metrics",
        f"- Liquidation value: ${liquidation:,.0f}",
        f"- Cash: ${cash:,.0f} ({(cash / liquidation) * 100:.1f}% of portfolio)",
        f"- CSP reserved cash: ${csp_reserved:,.0f}",
        f"- Cash after CSP reserves: ${cash_after_csp:,.0f}",
        f"- Suggested min cash buffer ({min_cash_buffer_pct:.0f}%): ${min_cash_buffer:,.0f}",
        f"- Deployable cash (after CSP + buffer): ${deployable_cash:,.0f}",
        f"- Distinct symbols (aggregated): {len(ranked)}",
        f"- Effective diversification (~1/HHI): {effective_names:.1f} names",
        f"- Top 1 / 3 / 5 weights: {top1:.1f}% / {top3:.1f}% / {top5:.1f}%",
        f"- Single-name target from profile: {single_name_limit:.0f}% max",
    ]

    lines.append("\n## Top holdings by weight")
    for symbol, market_value, weight in ranked[:10]:
        flag = _concentration_flags(weight, single_name_limit)
        suffix = f" — {flag}" if flag else ""
        lines.append(f"- {symbol}: {weight:.1f}% (${market_value:,.0f}){suffix}")

    if sector_weights:
        lines.append("\n## Sector weights (from research snapshot)")
        for sector_weight in sector_weights[:8]:
            sector = normalize_sector_label(sector_weight.sector)
            symbols = ", ".join(sector_weight.symbols[:5])
            flag = ""
            if sector_weight.weight_pct >= 30:
                flag = " — SECTOR CRITICAL (>30%)"
            elif sector_weight.weight_pct >= 25:
                flag = " — SECTOR HIGH (>25%)"
            lines.append(
                f"- {sector}: {sector_weight.weight_pct:.1f}% ({symbols}){flag}"
            )

    short_puts = _short_put_underlyings(positions)
    if short_puts:
        lines.append(
            "\n## Options overlay (short puts)"
            f"\nCash-secured put underlyings: {', '.join(short_puts)}"
            "\nTreat these as intentional concentration — avoid adding size unless underweight."
        )

    if (
        profile
        and profile.primary_strategy == InvestmentStrategy.ETF_CORE
        and profile.etf_core
        and profile.etf_core.target_allocation
    ):
        alloc = profile.etf_core.target_allocation
        weight_by_symbol = {symbol: weight for symbol, _, weight in ranked}
        value_by_symbol = _market_value_by_symbol(ranked)
        etf_symbols = [symbol.upper() for symbol in alloc.keys()]
        etf_total_pct = sum(weight_by_symbol.get(symbol, 0.0) for symbol in etf_symbols)
        etf_total_value = sum(value_by_symbol.get(symbol, 0.0) for symbol in etf_symbols)
        non_etf_pct = max(100.0 - etf_total_pct, 0.0)

        lines.append("\n## ETF core weight in portfolio")
        lines.append(
            f"- Total in ETF core targets: {etf_total_pct:.1f}% of portfolio "
            f"(${etf_total_value:,.0f})"
        )
        lines.append(
            f"- Non-ETF-core holdings (single names, options overlay, etc.): "
            f"{non_etf_pct:.1f}%"
        )
        for symbol, target_pct in alloc.items():
            symbol_upper = symbol.upper()
            current_pct = weight_by_symbol.get(symbol_upper, 0.0)
            current_value = value_by_symbol.get(symbol_upper, 0.0)
            lines.append(
                f"  - {symbol_upper}: {current_pct:.1f}% of portfolio "
                f"(${current_value:,.0f}) vs {target_pct:.0f}% target"
            )

        if etf_fund_metrics:
            lines.append("\n## ETF fund metrics (yield and fees)")
            for symbol in alloc.keys():
                symbol_upper = symbol.upper()
                metrics = etf_fund_metrics.get(symbol_upper) or {}
                yield_text = metrics.get("dividend_yield") or "N/A"
                expense_text = metrics.get("expense_ratio") or "N/A"
                lines.append(
                    f"- {symbol_upper}: dividend yield {yield_text}, "
                    f"expense ratio {expense_text}"
                )

        lines.append("\n## ETF core allocation gap (current vs target)")
        total_buy_gap = 0.0
        underweight_gaps: list[tuple[str, float]] = []
        for symbol, target_pct in alloc.items():
            symbol_upper = symbol.upper()
            current_pct = weight_by_symbol.get(symbol_upper, 0.0)
            gap_pct = target_pct - current_pct
            buy_dollars = max((gap_pct / 100.0) * liquidation, 0.0)
            total_buy_gap += buy_dollars
            status = "UNDERWEIGHT" if gap_pct > 1.0 else "on target" if abs(gap_pct) <= 1.0 else "OVERWEIGHT"
            lines.append(
                f"- {symbol_upper}: {current_pct:.1f}% now vs {target_pct:.0f}% target "
                f"({gap_pct:+.1f} pp) — {status}"
                + (f", ~${buy_dollars:,.0f} to buy" if buy_dollars >= 1.0 else "")
            )
            if gap_pct > 1.0 and buy_dollars >= 1.0:
                underweight_gaps.append((symbol_upper, buy_dollars))
        if deployable_cash > 0 and total_buy_gap > 0:
            deploy_pct = min(deployable_cash / total_buy_gap, 1.0) * 100.0
            lines.append(
                f"- Deployable cash ${deployable_cash:,.0f} covers "
                f"{deploy_pct:.0f}% of total ETF underweight gap (~${total_buy_gap:,.0f})"
            )
            if deployable_cash < total_buy_gap:
                lines.append(
                    "- Prioritize closing the largest ETF gaps first with available deployable cash; "
                    "use only the tickers listed above. Note any remaining gap for future contributions."
                )
            if underweight_gaps:
                total_underweight_gap = sum(gap for _, gap in underweight_gaps)
                lines.append(
                    "\n## Suggested deploy plan (precomputed from deployable cash + ETF gaps)"
                )
                for symbol_upper, gap_dollars in underweight_gaps:
                    deploy_amount = deployable_cash * (gap_dollars / total_underweight_gap)
                    lines.append(
                        f"- {symbol_upper}: ${deploy_amount:,.0f} "
                        f"(of ${deployable_cash:,.0f} deployable cash)"
                    )

    if profile and profile.primary_strategy == InvestmentStrategy.DIVIDEND:
        symbols = profile.dividend.dividend_symbols if profile.dividend else []
        if symbols:
            lines.append(
                "\n## Dividend watchlist\n"
                + ", ".join(symbols)
            )

    return "\n".join(lines)
