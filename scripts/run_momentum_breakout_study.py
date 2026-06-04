#!/usr/bin/env python3
"""One-off runner for Momentum Breakout research study (analysis only)."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.loader import load_symbol
from trade_planner.config import MomentumBreakoutConfig
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.feature_analysis import analyze_feature_conditions
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import MarketRegime
from trade_planner.research.regime_analysis import build_regime_comparison
from trade_planner.research.report_generator import (
    StrategyResearchReportGenerator,
    SymbolBarSet,
)
from trade_planner.research.data import align_benchmark_to_stock, ohlcv_bars_from_dataframe
from trade_planner.research.yearly import yearly_performance_table
from trade_planner.research.walk_forward import WalkForwardValidator

SYMBOLS = ("AAPL", "MSFT", "NVDA", "META", "AMZN")
BENCHMARK = "SPY"
REQUESTED_START = date(2000, 1, 1)
REQUESTED_END = date(2024, 12, 31)
CONFIG = MomentumBreakoutConfig()


def load_universe() -> tuple[list[SymbolBarSet], date, date, dict[str, date]]:
    bench_df = load_symbol(BENCHMARK)
    bench_all = ohlcv_bars_from_dataframe(bench_df)
    universe: list[SymbolBarSet] = []
    first_dates: dict[str, date] = {}
    effective_start = REQUESTED_END

    for symbol in SYMBOLS:
        df = load_symbol(symbol)
        sym_first = df.index.min().date()
        first_dates[symbol] = sym_first
        start_ts = pd.Timestamp(max(REQUESTED_START, sym_first))
        end_ts = pd.Timestamp(REQUESTED_END)
        sliced = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        stock_bars = ohlcv_bars_from_dataframe(sliced)
        if not stock_bars:
            continue
        effective_start = min(effective_start, stock_bars[0].trading_date)
        bench_slice = tuple(
            b for b in bench_all if stock_bars[0].trading_date <= b.trading_date <= REQUESTED_END
        )
        aligned = align_benchmark_to_stock(stock_bars, bench_slice)
        universe.append(
            SymbolBarSet(symbol=symbol, stock_bars=stock_bars, benchmark_bars=aligned)
        )

    return universe, effective_start, REQUESTED_END, first_dates


def combo_analysis(
    trades: tuple[HistoricalTrade, ...],
    *,
    top_n: int = 10,
) -> tuple[list[dict], list[dict]]:
    """2D threshold grids for RS × volume × close_vs_sma50 (study script only)."""
    grids = [
        ("rs_percentile", [80, 85, 90, 95]),
        ("volume_ratio", [1.5, 2.0, 2.5, 3.0]),
        ("close_vs_sma50", [0.0, 0.02, 0.05, 0.08]),
    ]

    def passes(trade: HistoricalTrade, rules: list[tuple[str, float]]) -> bool:
        snap = trade.feature_snapshot
        if snap is None:
            return False
        for name, threshold in rules:
            val = getattr(snap, name, None)
            if val is None or val < threshold:
                return False
        return True

    combos: list[dict] = []
    for (n1, t1s), (n2, t2s) in product(grids[:2], grids[1:3]):
        for t1 in t1s:
            for t2 in t2s:
                rules = [(n1, t1), (n2, t2)]
                bucket = tuple(t for t in trades if passes(t, rules))
                if len(bucket) < 5:
                    continue
                perf = performance_from_trades(bucket)
                combos.append(
                    {
                        "label": f"{n1} >= {t1}, {n2} >= {t2}",
                        "rules": rules,
                        "count": len(bucket),
                        "expectancy": perf.expectancy,
                        "win_rate": perf.win_rate,
                        "profit_factor": perf.profit_factor,
                    }
                )

    for (n1, t1s), (n2, t2s) in product([grids[0]], [grids[2]]):
        for t1 in t1s:
            for t2 in t2s:
                rules = [(n1, t1), (n2, t2)]
                bucket = tuple(t for t in trades if passes(t, rules))
                if len(bucket) < 5:
                    continue
                perf = performance_from_trades(bucket)
                combos.append(
                    {
                        "label": f"{n1} >= {t1}, close_vs_sma50 >= {t2:.0%}",
                        "rules": rules,
                        "count": len(bucket),
                        "expectancy": perf.expectancy,
                        "win_rate": perf.win_rate,
                        "profit_factor": perf.profit_factor,
                    }
                )

    ranked = sorted(combos, key=lambda row: row["expectancy"], reverse=True)
    top = ranked[:top_n]
    bottom = ranked[-top_n:][::-1] if len(ranked) >= top_n else ranked[:0]
    if len(ranked) < top_n:
        bottom = sorted(combos, key=lambda row: row["expectancy"])[:top_n]
    return top, bottom


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_pf(x: float) -> str:
    return "∞" if x >= 999 else f"{x:.2f}"


def main() -> None:
    universe, eff_start, eff_end, first_dates = load_universe()
    generator = StrategyResearchReportGenerator()
    trades = generator.collect_trades(universe, start_date=eff_start, end_date=eff_end)
    report = generator.generate(
        universe, start_date=eff_start, end_date=eff_end, trades=trades
    )
    yearly = yearly_performance_table(trades, setup_name=report.setup_name)
    wf = report.walk_forward
    regime = report.regime_comparison
    top_single, worst_single = analyze_feature_conditions(trades, top_n=10)
    top_combo, bottom_combo = combo_analysis(trades, top_n=10)

    o = report.performance
    lines: list[str] = []
    lines.append("# Momentum Breakout — Full Research Study")
    lines.append("")
    lines.append("**Universe:** AAPL, MSFT, NVDA, META, AMZN  ")
    lines.append(f"**Requested period:** {REQUESTED_START} → {REQUESTED_END}  ")
    lines.append(f"**Effective sample start:** {eff_start} (limited by listing history)  ")
    lines.append("")
    lines.append("## Data availability")
    lines.append("")
    lines.append("| Symbol | First bar in dataset |")
    lines.append("|--------|----------------------|")
    for sym in SYMBOLS:
        lines.append(f"| {sym} | {first_dates.get(sym, 'N/A')} |")
    lines.append("")
    lines.append(f"**Total historical trades:** {len(trades)}  ")
    lines.append(f"**Current production config:** RS ≥ {CONFIG.rs_percentile_min}, volume ratio ≥ {CONFIG.volume_ratio_min}, target R = {CONFIG.target_risk_reward}, stop = {CONFIG.stop_lookback_days}-day low  ")
    lines.append("")

    lines.append("## 1. Overall performance")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total trades | {o.total_trades} |")
    lines.append(f"| Win rate | {pct(o.win_rate)} |")
    lines.append(f"| Average win | {pct(o.average_win)} |")
    lines.append(f"| Average loss | {pct(o.average_loss)} |")
    lines.append(f"| Expectancy | {pct(o.expectancy)} |")
    lines.append(f"| Profit factor | {fmt_pf(o.profit_factor)} |")
    lines.append(f"| Sharpe ratio | {o.sharpe_ratio:.2f} |")
    lines.append(f"| Max drawdown | {pct(o.max_drawdown)} |")
    lines.append(f"| Average holding days | {o.average_holding_days:.1f} |")
    lines.append("")

    lines.append("## 2. Walk-forward (out-of-sample years)")
    lines.append("")
    lines.append("| Test year | Trades | Win rate | Profit factor | Expectancy | Flags |")
    lines.append("|-----------|--------|----------|---------------|------------|-------|")
    flagged_years: list[int] = []
    for fold in wf.folds:
        p = fold.performance
        flags = []
        if p.profit_factor < 1.0:
            flags.append("PF < 1")
        if p.expectancy < 0:
            flags.append("E < 0")
        if flags:
            flagged_years.append(fold.test_year)
        flag_str = ", ".join(flags) if flags else "—"
        lines.append(
            f"| {fold.test_year} | {p.total_trades} | {pct(p.win_rate)} | {fmt_pf(p.profit_factor)} | {pct(p.expectancy)} | {flag_str} |"
        )
    agg = wf.aggregate
    lines.append(
        f"| **OOS aggregate** | **{agg.total_trades}** | **{pct(agg.win_rate)}** | **{fmt_pf(agg.profit_factor)}** | **{pct(agg.expectancy)}** | — |"
    )
    lines.append("")
    if flagged_years:
        lines.append(
            f"**Flagged OOS years (PF < 1 or negative expectancy):** {', '.join(str(y) for y in flagged_years)}"
        )
    else:
        lines.append("**No OOS years flagged** — all test windows show PF ≥ 1 and non-negative expectancy.")
    lines.append("")

    lines.append("## 3. Regime analysis")
    lines.append("")
    lines.append("| Regime | Trades | Win rate | Profit factor | Expectancy | Avg return |")
    lines.append("|--------|--------|----------|---------------|------------|------------|")
    regime_rows = {row.regime: row.performance for row in regime.rows}
    disable: list[str] = []
    for reg in (MarketRegime.RISK_ON, MarketRegime.NEUTRAL, MarketRegime.RISK_OFF):
        p = regime_rows[reg]
        lines.append(
            f"| {reg.value} | {p.total_trades} | {pct(p.win_rate)} | {fmt_pf(p.profit_factor)} | {pct(p.expectancy)} | {pct(p.average_return)} |"
        )
        if p.total_trades >= 10 and (p.profit_factor < 1.0 or p.expectancy < 0):
            disable.append(reg.value)
    lines.append("")
    if disable:
        lines.append(
            f"**Regime filter recommendation:** Consider disabling entries in **{', '.join(disable)}** (PF < 1 or negative expectancy with ≥10 trades)."
        )
    else:
        lines.append(
            "**Regime filter recommendation:** No regime meets the disable threshold (PF < 1 and negative expectancy with ≥10 trades). Prefer sizing down in weaker regimes rather than a hard off switch."
        )
    lines.append("")

    lines.append("## 4. Feature analysis")
    lines.append("")
    lines.append("### Top 10 single-feature bins (by expectancy)")
    lines.append("")
    lines.append("| Rank | Feature | Range | Trades | Win rate | Expectancy |")
    lines.append("|------|---------|-------|--------|----------|------------|")
    for idx, row in enumerate(top_single, 1):
        lines.append(
            f"| {idx} | {row.feature} | {row.range_label} | {row.trade_count} | {pct(row.win_rate)} | {pct(row.expectancy)} |"
        )
    lines.append("")
    lines.append("### Bottom 10 single-feature bins")
    lines.append("")
    lines.append("| Rank | Feature | Range | Trades | Win rate | Expectancy |")
    lines.append("|------|---------|-------|--------|----------|------------|")
    for idx, row in enumerate(worst_single, 1):
        lines.append(
            f"| {idx} | {row.feature} | {row.range_label} | {row.trade_count} | {pct(row.win_rate)} | {pct(row.expectancy)} |"
        )
    lines.append("")
    lines.append("### Top 10 multi-feature combinations (RS × volume × trend proxies)")
    lines.append("")
    lines.append("| Rank | Condition | Trades | Win rate | PF | Expectancy |")
    lines.append("|------|-----------|--------|----------|-----|------------|")
    for idx, row in enumerate(top_combo, 1):
        lines.append(
            f"| {idx} | {row['label']} | {row['count']} | {pct(row['win_rate'])} | {fmt_pf(row['profit_factor'])} | {pct(row['expectancy'])} |"
        )
    lines.append("")
    lines.append("### Bottom 10 multi-feature combinations")
    lines.append("")
    lines.append("| Rank | Condition | Trades | Win rate | PF | Expectancy |")
    lines.append("|------|-----------|--------|----------|-----|------------|")
    for idx, row in enumerate(bottom_combo, 1):
        lines.append(
            f"| {idx} | {row['label']} | {row['count']} | {pct(row['win_rate'])} | {fmt_pf(row['profit_factor'])} | {pct(row['expectancy'])} |"
        )
    lines.append("")

    lines.append("## 5. Yearly in-sample stability")
    lines.append("")
    lines.append("| Year | Trades | Win rate | PF | Expectancy |")
    lines.append("|------|--------|----------|-----|------------|")
    for row in yearly:
        p = row.performance
        lines.append(
            f"| {row.year} | {p.total_trades} | {pct(p.win_rate)} | {fmt_pf(p.profit_factor)} | {pct(p.expectancy)} |"
        )
    lines.append("")

    # Recommendations
    edge_positive = o.expectancy > 0 and o.profit_factor > 1.0 and o.total_trades >= 30
    oos_stable = agg.expectancy > 0 and agg.profit_factor > 1.0
    oos_pct_good = sum(
        1
        for f in wf.folds
        if f.performance.profit_factor >= 1.0 and f.performance.expectancy >= 0
    ) / max(len(wf.folds), 1)

    risk_on = regime_rows[MarketRegime.RISK_ON]
    risk_off = regime_rows[MarketRegime.RISK_OFF]
    regime_dependent = (
        risk_on.total_trades >= 5
        and risk_off.total_trades >= 5
        and risk_on.expectancy > risk_off.expectancy + 0.01
    )

    lines.append("## 6. Recommendation report")
    lines.append("")
    lines.append("### Does Momentum Breakout have a positive edge?")
    if edge_positive:
        lines.append(
            f"- **Yes (aggregate):** Expectancy {pct(o.expectancy)}, profit factor {fmt_pf(o.profit_factor)}, Sharpe {o.sharpe_ratio:.2f} over {o.total_trades} trades."
        )
    else:
        lines.append(
            f"- **Unclear / weak (aggregate):** Expectancy {pct(o.expectancy)}, profit factor {fmt_pf(o.profit_factor)} on {o.total_trades} trades — does not clear a simple PF > 1 and E > 0 rule."
        )
    lines.append("")
    lines.append("### Is the edge stable across years?")
    lines.append(
        f"- OOS aggregate ({agg.total_trades} trades): expectancy {pct(agg.expectancy)}, PF {fmt_pf(agg.profit_factor)}."
    )
    lines.append(
        f"- {oos_pct_good * 100:.0f}% of walk-forward test years ({sum(1 for f in wf.folds if f.performance.profit_factor >= 1.0 and f.performance.expectancy >= 0)}/{len(wf.folds)}) show PF ≥ 1 and non-negative expectancy."
    )
    if flagged_years:
        lines.append(f"- **Instability:** Flagged years: {', '.join(str(y) for y in flagged_years)}.")
    lines.append("")
    lines.append("### Is the edge regime dependent?")
    if regime_dependent:
        lines.append(
            f"- **Yes:** RISK_ON expectancy {pct(risk_on.expectancy)} vs RISK_OFF {pct(risk_off.expectancy)} ({risk_on.total_trades} vs {risk_off.total_trades} trades)."
        )
    else:
        lines.append("- **Mixed / insufficient separation** between regimes on this sample.")
    lines.append("")
    lines.append("### Feature ranges that improve expectancy")
    if top_single:
        best = top_single[0]
        lines.append(
            f"- Best single-feature bin: **{best.feature}** in [{best.range_label}] (E={pct(best.expectancy)}, n={best.trade_count})."
        )
    if top_combo:
        lines.append(f"- Best combo: **{top_combo[0]['label']}** (E={pct(top_combo[0]['expectancy'])}, n={top_combo[0]['count']}).")
    lines.append("")
    lines.append("### Feature ranges to filter out")
    if worst_single:
        worst = worst_single[0]
        lines.append(
            f"- Worst single-feature bin: **{worst.feature}** in [{worst.range_label}] (E={pct(worst.expectancy)}, n={worst.trade_count})."
        )
    if bottom_combo:
        lines.append(
            f"- Worst combo: **{bottom_combo[0]['label']}** (E={pct(bottom_combo[0]['expectancy'])}, n={bottom_combo[0]['count']})."
        )
    lines.append("")

    lines.append("## 7. Proposed configuration (data-driven, no code changes applied)")
    lines.append("")
    rs_suggest = 85.0
    vol_suggest = 1.75
    if top_combo:
        for name, thr in top_combo[0]["rules"]:
            if name == "rs_percentile":
                rs_suggest = max(rs_suggest, thr)
            if name == "volume_ratio":
                vol_suggest = max(vol_suggest, thr)
    regime_filter = "RISK_ON only" if disable == ["RISK_OFF"] else (
        "Disable RISK_OFF" if MarketRegime.RISK_OFF.value in disable else "No hard regime gate; reduce size in NEUTRAL/RISK_OFF"
    )
    lines.append("| Parameter | Current | Proposed | Rationale |")
    lines.append("|-----------|---------|----------|-----------|")
    lines.append(f"| RS percentile min | {CONFIG.rs_percentile_min} | **{rs_suggest:.0f}** | Lift threshold toward top-expectancy RS bins |")
    lines.append(f"| Volume ratio min | {CONFIG.volume_ratio_min} | **{vol_suggest:.2f}** | Emphasize expansion days linked to winners |")
    lines.append(f"| Stop | {CONFIG.stop_lookback_days}-day low | **Keep 10-day low** | Standard rule; no evidence here to widen |")
    lines.append(f"| Target R | {CONFIG.target_risk_reward} | **Keep 2.0R** | Fixed plan geometry unless PF < 1 drives retest |")
    lines.append(f"| Regime filter | None | **{regime_filter}** | From regime table |")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/run_momentum_breakout_study.py` using existing `trade_planner.research` pipeline.*")

    out_path = ROOT / "docs" / "momentum_breakout_research_study.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(out_path)
    print(f"trades={len(trades)} expectancy={o.expectancy:.4f} pf={o.profit_factor:.2f}")


if __name__ == "__main__":
    main()
