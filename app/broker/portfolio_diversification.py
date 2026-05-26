from __future__ import annotations

from typing import TypedDict

from app.broker.option_utils import total_csp_reserved_cash
from app.broker.position_metrics import portfolio_liquidation_value
from app.broker.sector_labels import normalize_sector_label
from app.models.intelligence_models import SectorWeight
from app.models.portfolio_analysis_precomputed_models import (
    CashMapStep,
    DeployPlanItem,
    HoldingAllocationReview,
    PortfolioAnalysisPrecomputed,
    PortfolioCashMap,
    PortfolioConcentrationMetrics,
    TrimPlanItem,
)
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
        return "Too large — over 30% of portfolio"
    if weight_pct >= 20:
        return "Very large — 20–30% of portfolio"
    if weight_pct >= 15:
        return "Large — 15–20% of portfolio"
    if weight_pct >= limit_pct:
        return f"Above your {limit_pct:.0f}% per-stock limit"
    return ""


def _trim_target_weight_pct(weight_pct: float, single_limit: float) -> float | None:
    if weight_pct >= 30:
        return 20.0
    if weight_pct >= 20:
        return 15.0
    if weight_pct >= single_limit:
        return max(single_limit - 1.0, 10.0)
    return None


def _trim_dollars_to_target(
    *,
    market_value: float,
    weight_pct: float,
    target_weight_pct: float,
    liquidation: float,
) -> float:
    if weight_pct <= target_weight_pct:
        return 0.0
    target_value = (target_weight_pct / 100.0) * liquidation
    return max(round(market_value - target_value, 2), 0.0)


def _holding_allocation_status(
    *,
    weight_pct: float,
    single_limit: float,
    etf_target_pct: float | None = None,
) -> str:
    flag = _concentration_flags(weight_pct, single_limit)
    if flag.startswith("Too large") or flag.startswith("Very large"):
        return flag
    if flag:
        return flag
    if etf_target_pct is not None:
        gap = etf_target_pct - weight_pct
        if gap > 1.0:
            return "Below ETF target"
        if gap < -1.0:
            return "Above ETF target"
        return "At ETF target"
    if weight_pct < single_limit * 0.5:
        return "Small position — room to add"
    return "Good size"


def _profile_single_name_limit(profile: UserInvestmentProfile | None) -> float:
    if profile and profile.wheel and profile.wheel.max_single_name_pct:
        return float(profile.wheel.max_single_name_pct)
    if profile and profile.risk_tolerance == "conservative":
        return 12.0
    if profile and profile.risk_tolerance == "aggressive":
        return 20.0
    return 15.0


def build_portfolio_allocation_precomputed(
    *,
    positions: list[Position],
    account: SchwabAccounts,
    profile: UserInvestmentProfile | None = None,
) -> PortfolioAnalysisPrecomputed | None:
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

    etf_targets: dict[str, float] = {}
    if (
        profile
        and profile.primary_strategy == InvestmentStrategy.ETF_CORE
        and profile.etf_core
        and profile.etf_core.target_allocation
    ):
        etf_targets = {
            symbol.upper(): target
            for symbol, target in profile.etf_core.target_allocation.items()
        }

    holdings: list[HoldingAllocationReview] = []
    trim_plan: list[TrimPlanItem] = []
    total_trim_proceeds = 0.0

    for symbol, market_value, weight in ranked[:12]:
        etf_target = etf_targets.get(symbol)
        status = _holding_allocation_status(
            weight_pct=weight,
            single_limit=single_name_limit,
            etf_target_pct=etf_target,
        )
        action_bits: list[str] = []
        trim_target = _trim_target_weight_pct(weight, single_name_limit)
        trim_dollars = 0.0
        if trim_target is not None:
            trim_dollars = _trim_dollars_to_target(
                market_value=market_value,
                weight_pct=weight,
                target_weight_pct=trim_target,
                liquidation=liquidation,
            )
            if trim_dollars >= 1.0:
                total_trim_proceeds += trim_dollars
                action_bits.append(
                    f"Sell about ${trim_dollars:,.0f} to bring this closer to {trim_target:.0f}% of your portfolio"
                )
                trim_plan.append(
                    TrimPlanItem(
                        symbol=symbol,
                        current_weight_pct=round(weight, 2),
                        target_weight_pct=trim_target,
                        trim_dollars=trim_dollars,
                    )
                )
        if etf_target is not None:
            gap_pct = etf_target - weight
            buy_dollars = max((gap_pct / 100.0) * liquidation, 0.0)
            if gap_pct > 1.0 and buy_dollars >= 1.0:
                action_bits.append(
                    f"Buy about ${buy_dollars:,.0f} more to reach your ETF target"
                )
            elif gap_pct < -1.0:
                action_bits.append(
                    "Above your ETF target — trim this before buying elsewhere"
                )
        if not action_bits:
            if weight >= 15:
                action_bits.append(
                    "Already a big slice of your portfolio — don't add more yet"
                )
            elif status == "Small position — room to add":
                action_bits.append(
                    f"Still small vs your {single_name_limit:.0f}% per-stock limit — OK to buy more if it fits your plan"
                )
            else:
                action_bits.append("Fine to hold — no trim needed")

        holdings.append(
            HoldingAllocationReview(
                symbol=symbol,
                weight_pct=round(weight, 2),
                market_value=round(market_value, 2),
                status=status,
                action_summary="; ".join(action_bits),
            )
        )

    total_to_redeploy = deployable_cash + total_trim_proceeds
    cash_steps: list[CashMapStep] = [
        CashMapStep(step=1, label="Cash in account", amount=round(cash, 2)),
        CashMapStep(
            step=2,
            label="Cash reserved for short puts",
            amount=round(csp_reserved, 2),
            is_subtraction=True,
        ),
        CashMapStep(
            step=3,
            label="Cash left after put reserves",
            amount=round(cash_after_csp, 2),
        ),
        CashMapStep(
            step=4,
            label=f"Emergency cash buffer ({min_cash_buffer_pct:.0f}% of portfolio)",
            amount=round(min_cash_buffer, 2),
            is_subtraction=True,
        ),
        CashMapStep(
            step=5,
            label="Cash you can invest today",
            amount=round(deployable_cash, 2),
        ),
    ]
    if total_trim_proceeds > 0:
        cash_steps.extend(
            [
                CashMapStep(
                    step=6,
                    label="If you trim oversized positions",
                    amount=round(total_trim_proceeds, 2),
                ),
                CashMapStep(
                    step=7,
                    label="Total cash available to invest",
                    amount=round(total_to_redeploy, 2),
                ),
            ]
        )

    deploy_plan: list[DeployPlanItem] = []
    if etf_targets:
        underweight_gaps: list[tuple[str, float]] = []
        weight_by_symbol = {symbol: weight for symbol, _, weight in ranked}
        for symbol, target_pct in etf_targets.items():
            current_pct = weight_by_symbol.get(symbol, 0.0)
            gap_pct = target_pct - current_pct
            buy_dollars = max((gap_pct / 100.0) * liquidation, 0.0)
            if gap_pct > 1.0 and buy_dollars >= 1.0:
                underweight_gaps.append((symbol, buy_dollars))
        if underweight_gaps and total_to_redeploy > 0:
            total_underweight_gap = sum(gap for _, gap in underweight_gaps)
            remaining = total_to_redeploy
            for index, (symbol_upper, gap_dollars) in enumerate(underweight_gaps):
                if index == len(underweight_gaps) - 1:
                    deploy_amount = remaining
                else:
                    deploy_amount = total_to_redeploy * (
                        gap_dollars / total_underweight_gap
                    )
                    remaining -= deploy_amount
                deploy_plan.append(
                    DeployPlanItem(
                        symbol=symbol_upper,
                        deploy_dollars=round(deploy_amount, 2),
                        note=f"About ${gap_dollars:,.0f} below your target allocation",
                    )
                )

    return PortfolioAnalysisPrecomputed(
        concentration=PortfolioConcentrationMetrics(
            liquidation_value=round(liquidation, 2),
            cash=round(cash, 2),
            cash_pct=round((cash / liquidation) * 100.0, 2),
            csp_reserved=round(csp_reserved, 2),
            cash_after_csp=round(cash_after_csp, 2),
            min_cash_buffer=round(min_cash_buffer, 2),
            deployable_cash=round(deployable_cash, 2),
            distinct_symbols=len(ranked),
            effective_names=round(effective_names, 2),
            top1_pct=round(top1, 2),
            top3_pct=round(top3, 2),
            top5_pct=round(top5, 2),
            single_name_limit_pct=single_name_limit,
        ),
        cash_map=PortfolioCashMap(
            steps=cash_steps,
            deployable_cash=round(deployable_cash, 2),
            trim_proceeds=round(total_trim_proceeds, 2) if total_trim_proceeds > 0 else None,
            total_to_redeploy=round(total_to_redeploy, 2),
            min_cash_buffer_pct=min_cash_buffer_pct,
        ),
        holdings=holdings,
        trim_plan=trim_plan,
        deploy_plan=deploy_plan,
        total_trim_proceeds=round(total_trim_proceeds, 2),
    )


def format_diversification_summary_block(
    *,
    positions: list[Position],
    account: SchwabAccounts,
    sector_weights: list[SectorWeight] | None = None,
    profile: UserInvestmentProfile | None = None,
    etf_fund_metrics: dict[str, EtfFundMetrics] | None = None,
) -> str | None:
    precomputed = build_portfolio_allocation_precomputed(
        positions=positions,
        account=account,
        profile=profile,
    )
    if precomputed is None:
        return None

    c = precomputed.concentration
    lines = [
        "Note: the user sees a money map card with cash buckets, holdings, and trim/deploy $. "
        "Use this block for analysis only — do not repeat it verbatim in JSON output.",
        "",
        "## Key numbers",
        f"- Cash you can invest today: ${c.deployable_cash:,.0f}",
        f"- CSP reserved cash: ${c.csp_reserved:,.0f}",
        f"- Max per-stock limit (from profile): {c.single_name_limit_pct:.0f}%",
        f"- Largest holding: {c.top1_pct:.1f}% of portfolio",
        f"- Top 3 holdings combined: {c.top3_pct:.1f}%",
        "",
        "## Portfolio cash map (precomputed — for reasoning only)",
    ]
    for step in precomputed.cash_map.steps:
        if step.amount is None:
            continue
        if step.is_subtraction:
            lines.append(f"{step.step}. {step.label}: −${step.amount:,.0f}")
        elif step.step in {5, 7}:
            lines.append(f"{step.step}. = {step.label}: ${step.amount:,.0f}")
        else:
            lines.append(f"{step.step}. {step.label}: ${step.amount:,.0f}")
    if precomputed.total_trim_proceeds <= 0:
        lines.append(
            "- No mandatory single-name trims precomputed — deployable cash is the main pool."
        )

    lines.append(
        "\n## Holding-by-holding review (precomputed — for reasoning only)"
    )
    for holding in precomputed.holdings:
        lines.append(
            f"- {holding.symbol}: {holding.weight_pct:.1f}% "
            f"(${holding.market_value:,.0f}) — {holding.status} — "
            f"{holding.action_summary}"
        )

    if precomputed.trim_plan:
        lines.append(
            "\n## Suggested trim plan (precomputed — rank before new buys when overweight)"
        )
        for item in precomputed.trim_plan:
            lines.append(
                f"- {item.symbol}: {item.current_weight_pct:.1f}% now → "
                f"trim ~${item.trim_dollars:,.0f} "
                f"(target ~{item.target_weight_pct:.0f}% of portfolio)"
            )
        lines.append(
            f"- Total trim proceeds if executed: ~${precomputed.total_trim_proceeds:,.0f}"
        )

    liquidation = c.liquidation_value
    ranked = _aggregate_symbol_weights(positions, liquidation)
    deployable_cash = c.deployable_cash
    total_trim_proceeds = precomputed.total_trim_proceeds

    if sector_weights:
        lines.append("\n## Sector weights (from research snapshot)")
        for sector_weight in sector_weights[:8]:
            sector = normalize_sector_label(sector_weight.sector)
            symbols = ", ".join(sector_weight.symbols[:5])
            flag = ""
            if sector_weight.weight_pct >= 30:
                flag = " — sector is over 30% of portfolio"
            elif sector_weight.weight_pct >= 25:
                flag = " — sector is heavily weighted (over 25%)"
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
                total_redeploy = deployable_cash + total_trim_proceeds
                lines.append(
                    "\n## Suggested deploy plan (precomputed from deployable cash + ETF gaps)"
                )
                if total_trim_proceeds > 0:
                    lines.append(
                        f"- Assume trims above free ~${total_trim_proceeds:,.0f}; "
                        f"total to allocate this month: ~${total_redeploy:,.0f}"
                    )
                for item in precomputed.deploy_plan:
                    note = f" ({item.note})" if item.note else ""
                    lines.append(f"- {item.symbol}: ${item.deploy_dollars:,.0f}{note}")
                if total_redeploy > total_underweight_gap:
                    lines.append(
                        f"- Remaining after closing ETF gaps: "
                        f"~${max(total_redeploy - total_underweight_gap, 0):,.0f} — hold as buffer "
                        "or add to largest underweight only if still below target."
                    )

    if profile and profile.primary_strategy == InvestmentStrategy.DIVIDEND:
        symbols = profile.dividend.dividend_symbols if profile.dividend else []
        if symbols:
            lines.append(
                "\n## Dividend watchlist\n"
                + ", ".join(symbols)
            )

    return "\n".join(lines)
