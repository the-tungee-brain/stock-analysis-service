from __future__ import annotations

from app.broker.option_utils import total_csp_reserved_cash
from app.broker.position_metrics import portfolio_liquidation_value
from app.broker.sector_labels import normalize_sector_label
from app.models.intelligence_models import SectorWeight
from app.models.schwab_models import Position, SchwabAccounts
from app.models.strategy_models import InvestmentStrategy, UserInvestmentProfile


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

    if profile and profile.etf_core and profile.etf_core.target_allocation:
        alloc = profile.etf_core.target_allocation
        alloc_text = ", ".join(f"{sym} {pct:.0f}%" for sym, pct in alloc.items())
        lines.append(f"\n## ETF core target allocation\n{alloc_text}")

    if profile and profile.primary_strategy == InvestmentStrategy.DIVIDEND:
        symbols = profile.dividend.dividend_symbols if profile.dividend else []
        if symbols:
            lines.append(
                "\n## Dividend watchlist\n"
                + ", ".join(symbols)
            )

    return "\n".join(lines)
