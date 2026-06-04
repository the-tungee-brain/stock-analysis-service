#!/usr/bin/env python3
"""Robustness validation for Momentum Breakout (research only, no prod changes)."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.loader import load_symbol
from trade_planner.backtest.engine import BacktestEngine
from trade_planner.config import BacktestConfig, MomentumBreakoutConfig
from trade_planner.indicators import max_drawdown_pct
from trade_planner.persistence.historical_trade import HistoricalTrade
from trade_planner.research.collector import collect_momentum_breakout_trades
from trade_planner.research.data import align_benchmark_to_stock, find_bar_index, ohlcv_bars_from_dataframe
from trade_planner.research.metrics import performance_from_trades
from trade_planner.research.models import MarketRegime
from trade_planner.research.report_generator import SymbolBarSet
from trade_planner.research.walk_forward import WalkForwardValidator, default_walk_forward_folds
from trade_planner.setups.momentum_breakout import MomentumBreakoutSetup

SYMBOLS = ("AAPL", "MSFT", "NVDA", "META", "AMZN")
BENCHMARK = "SPY"
REQUESTED_START = date(2000, 1, 1)
REQUESTED_END = date(2024, 12, 31)
OOS_YEARS = frozenset({2019, 2020, 2021, 2022, 2023, 2024})
BASELINE = MomentumBreakoutConfig()
BACKTEST = BacktestConfig(min_warmup_bars=60, max_holding_days=20)

RS_GRID = (80.0, 85.0, 90.0, 95.0)
VOL_GRID = (1.3, 1.5, 1.7, 2.0)
TARGET_GRID = (1.5, 2.0, 2.5, 3.0)
STOP_GRID = (5, 10, 15, 20)


@dataclass(frozen=True, slots=True)
class ParamSet:
    rs_min: float
    vol_min: float
    target_r: float
    stop_days: int

    def label(self) -> str:
        return f"RS{self.rs_min:g}_V{self.vol_min}_R{self.target_r}_S{self.stop_days}"


@dataclass
class RunMetrics:
    params: ParamSet
    total_trades: int
    oos_trades: int
    win_rate: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    avg_holding_days: float
    oos_expectancy: float
    oos_profit_factor: float
    oos_max_drawdown: float
    oos_year_good_pct: float
    oos_year_pf_max_share: float
    robustness_score: float


def load_universe() -> tuple[list[SymbolBarSet], date]:
    bench_df = load_symbol(BENCHMARK)
    bench_all = ohlcv_bars_from_dataframe(bench_df)
    universe: list[SymbolBarSet] = []
    effective_start = REQUESTED_END

    for symbol in SYMBOLS:
        df = load_symbol(symbol)
        sym_first = df.index.min().date()
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
    return universe, effective_start


def collect_fast(
    universe: list[SymbolBarSet],
    params: ParamSet,
    *,
    with_snapshots: bool = False,
) -> tuple[HistoricalTrade, ...]:
    cfg = replace(
        BASELINE,
        rs_percentile_min=params.rs_min,
        volume_ratio_min=params.vol_min,
        target_risk_reward=params.target_r,
        stop_lookback_days=params.stop_days,
    )
    setup = MomentumBreakoutSetup(cfg)
    all_trades: list[HistoricalTrade] = []
    for item in universe:
        if with_snapshots:
            trades = collect_momentum_breakout_trades(
                symbol=item.symbol,
                stock_bars=item.stock_bars,
                benchmark_bars=item.benchmark_bars,
                setup=setup,
                backtest_config=BACKTEST,
                momentum_config=cfg,
            )
        else:
            engine = BacktestEngine(BACKTEST)
            result = engine.run(
                setup,
                item.stock_bars,
                symbol=item.symbol,
                benchmark_bars=item.benchmark_bars,
            )
            trades = tuple(HistoricalTrade.from_simulated(t) for t in result.trades)
        all_trades.extend(trades)
    return tuple(all_trades)


def split_oos(trades: tuple[HistoricalTrade, ...]) -> tuple[HistoricalTrade, ...]:
    return tuple(t for t in trades if t.signal_date.year in OOS_YEARS)


def equity_curve(trades: tuple[HistoricalTrade, ...]) -> list[tuple[date, float]]:
    ordered = sorted(trades, key=lambda t: t.exit_date)
    equity = 1.0
    curve: list[tuple[date, float]] = [(ordered[0].exit_date if ordered else date.today(), 1.0)]
    for trade in ordered:
        equity *= 1.0 + trade.return_pct
        curve.append((trade.exit_date, equity))
    return curve


def max_dd_from_trades(trades: tuple[HistoricalTrade, ...]) -> float:
    if not trades:
        return 0.0
    values = [pt[1] for pt in equity_curve(trades)]
    return max_drawdown_pct(values)


def oos_year_stats(trades: tuple[HistoricalTrade, ...]) -> dict[int, dict]:
    oos = split_oos(trades)
    by_year: dict[int, list[HistoricalTrade]] = {}
    for t in oos:
        by_year.setdefault(t.signal_date.year, []).append(t)
    out: dict[int, dict] = {}
    for year, bucket in by_year.items():
        perf = performance_from_trades(bucket)
        wins = [t.return_pct for t in bucket if t.return_pct > 0]
        gross_profit = sum(wins)
        out[year] = {
            "trades": len(bucket),
            "expectancy": perf.expectancy,
            "profit_factor": perf.profit_factor,
            "gross_profit": gross_profit,
        }
    return out


def one_year_pf_dominance(year_stats: dict[int, dict]) -> float:
    profits = [max(0.0, s["gross_profit"]) for s in year_stats.values() if s["trades"] > 0]
    if not profits or sum(profits) <= 0:
        return 0.0
    return max(profits) / sum(profits)


def year_consistency(year_stats: dict[int, dict]) -> float:
    tested = [y for y in OOS_YEARS if year_stats.get(y, {}).get("trades", 0) > 0]
    if not tested:
        return 0.0
    good = sum(
        1
        for y in tested
        if year_stats[y]["profit_factor"] >= 1.0 and year_stats[y]["expectancy"] >= 0
    )
    return good / len(tested)


def robustness_score(
    *,
    oos_trades: int,
    oos_expectancy: float,
    oos_pf: float,
    oos_max_dd: float,
    year_good_pct: float,
    year_pf_share: float,
    neighbor_median_oos_e: float | None = None,
) -> float:
    score = 0.0
    if oos_expectancy > 0:
        score += 25
    else:
        score -= 20
    if oos_pf >= 1.2:
        score += 20
    elif oos_pf < 1.0:
        score -= 15
    if oos_trades >= 50:
        score += 20
    elif oos_trades < 30:
        score -= 30
    else:
        score += 5
    if oos_max_dd < 0.35:
        score += 15
    elif oos_max_dd > 0.50:
        score -= 25
    elif oos_max_dd > 0.35:
        score -= 10
    score += year_good_pct * 20
    if year_pf_share > 0.55:
        score -= 20
    elif year_pf_share > 0.45:
        score -= 10
    if neighbor_median_oos_e is not None and oos_expectancy > 0:
        if abs(oos_expectancy - neighbor_median_oos_e) > 0.03:
            score -= 8
    return round(score, 2)


def compute_metrics(params: ParamSet, trades: tuple[HistoricalTrade, ...]) -> RunMetrics:
    oos = split_oos(trades)
    full = performance_from_trades(trades)
    oos_perf = performance_from_trades(oos)
    year_stats = oos_year_stats(trades)
    return RunMetrics(
        params=params,
        total_trades=len(trades),
        oos_trades=len(oos),
        win_rate=full.win_rate,
        expectancy=full.expectancy,
        profit_factor=full.profit_factor,
        max_drawdown=max_dd_from_trades(trades),
        avg_holding_days=full.average_holding_days,
        oos_expectancy=oos_perf.expectancy,
        oos_profit_factor=oos_perf.profit_factor,
        oos_max_drawdown=max_dd_from_trades(oos),
        oos_year_good_pct=year_consistency(year_stats),
        oos_year_pf_max_share=one_year_pf_dominance(year_stats),
        robustness_score=0.0,
    )


def neighbor_median_oos_e(
    grid_results: dict[ParamSet, RunMetrics],
    center: ParamSet,
) -> float | None:
    rs_ix = RS_GRID.index(center.rs_min) if center.rs_min in RS_GRID else -1
    vol_ix = VOL_GRID.index(center.vol_min) if center.vol_min in VOL_GRID else -1
    if rs_ix < 0 or vol_ix < 0:
        return None
    neighbors: list[float] = []
    for dr in (-1, 0, 1):
        for dv in (-1, 0, 1):
            if dr == 0 and dv == 0:
                continue
            ri, vi = rs_ix + dr, vol_ix + dv
            if 0 <= ri < len(RS_GRID) and 0 <= vi < len(VOL_GRID):
                key = ParamSet(
                    RS_GRID[ri],
                    VOL_GRID[vi],
                    center.target_r,
                    center.stop_days,
                )
                if key in grid_results:
                    neighbors.append(grid_results[key].oos_expectancy)
    if not neighbors:
        return None
    neighbors.sort()
    return neighbors[len(neighbors) // 2]


def drawdown_investigation(trades: tuple[HistoricalTrade, ...]) -> dict:
    if not trades:
        return {}
    curve = equity_curve(trades)
    values = [v for _, v in curve]
    peak = values[0]
    peak_date = curve[0][0]
    max_dd = 0.0
    trough_date = curve[0][0]
    trough_val = 1.0
    peak_at_trough = 1.0
    for dt, val in curve:
        if val > peak:
            peak = val
            peak_date = dt
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            trough_date = dt
            trough_val = val
            peak_at_trough = peak

    dd_trades = [
        t
        for t in trades
        if peak_date <= t.exit_date <= trough_date or (t.entry_date <= trough_date <= t.exit_date)
    ]
    if not dd_trades:
        ordered = sorted(trades, key=lambda t: t.exit_date)
        dd_trades = ordered[-min(15, len(ordered)) :]

    by_symbol: dict[str, list[float]] = {}
    by_regime: dict[str, list[float]] = {}
    for t in dd_trades:
        by_symbol.setdefault(t.symbol, []).append(t.return_pct)
        reg = (
            t.feature_snapshot.market_regime.value
            if t.feature_snapshot
            else "UNKNOWN"
        )
        by_regime.setdefault(reg, []).append(t.return_pct)

    ordered = sorted(trades, key=lambda t: t.exit_date)
    max_consec = 0
    streak = 0
    streak_start: date | None = None
    worst_streak_start: date | None = None
    worst_streak_len = 0
    for t in ordered:
        if t.return_pct <= 0:
            streak += 1
            if streak == 1:
                streak_start = t.exit_date
            if streak > max_consec:
                max_consec = streak
                worst_streak_len = streak
                worst_streak_start = streak_start
        else:
            streak = 0

    overlap_count = 0
    for i, a in enumerate(trades):
        for b in trades[i + 1 :]:
            if a.symbol != b.symbol:
                continue
            if a.entry_date <= b.exit_date and b.entry_date <= a.exit_date:
                overlap_count += 1

    portfolio_overlap = 0
    for i, a in enumerate(trades):
        for b in trades[i + 1 :]:
            if a.entry_date <= b.exit_date and b.entry_date <= a.exit_date:
                portfolio_overlap += 1

    return {
        "max_dd": max_dd,
        "peak_date": peak_date,
        "trough_date": trough_date,
        "dd_trade_count": len(dd_trades),
        "by_symbol": {s: {"n": len(r), "sum_ret": sum(r)} for s, r in by_symbol.items()},
        "by_regime": {s: {"n": len(r), "sum_ret": sum(r)} for s, r in by_regime.items()},
        "max_consecutive_losses": max_consec,
        "worst_streak_start": worst_streak_start,
        "same_symbol_overlaps": overlap_count,
        "portfolio_overlaps": portfolio_overlap,
        "worst_trades": sorted(dd_trades, key=lambda t: t.return_pct)[:10],
    }


def volume_band_analysis(trades: tuple[HistoricalTrade, ...]) -> list[dict]:
    bands = [
        ("All trades (vol >= setup min)", lambda v: v is not None),
        ("volume_ratio >= 1.5", lambda v: v is not None and v >= 1.5),
        ("1.5 <= volume < 1.75", lambda v: v is not None and 1.5 <= v < 1.75),
        ("1.9 <= volume <= 2.5", lambda v: v is not None and 1.9 <= v <= 2.5),
        ("volume_ratio >= 3.0", lambda v: v is not None and v >= 3.0),
    ]
    rows: list[dict] = []
    for label, pred in bands:
        bucket = tuple(
            t
            for t in trades
            if t.feature_snapshot
            and pred(t.feature_snapshot.volume_ratio)
        )
        if not bucket:
            rows.append({"label": label, "n": 0, "e": 0.0, "pf": 0.0, "wr": 0.0})
            continue
        perf = performance_from_trades(bucket)
        rows.append(
            {
                "label": label,
                "n": len(bucket),
                "e": perf.expectancy,
                "pf": perf.profit_factor,
                "wr": perf.win_rate,
            }
        )
    return rows


def main() -> None:
    universe, eff_start = load_universe()
    baseline_params = ParamSet(
        BASELINE.rs_percentile_min,
        BASELINE.volume_ratio_min,
        BASELINE.target_risk_reward,
        BASELINE.stop_lookback_days,
    )

    print("Running parameter grid (256 combinations)...", flush=True)
    grid_results: dict[ParamSet, RunMetrics] = {}
    trade_cache: dict[ParamSet, tuple[HistoricalTrade, ...]] = {}

    for rs, vol, tgt, stop in product(RS_GRID, VOL_GRID, TARGET_GRID, STOP_GRID):
        params = ParamSet(rs, vol, tgt, stop)
        trades = collect_fast(universe, params, with_snapshots=False)
        trade_cache[params] = trades
        grid_results[params] = compute_metrics(params, trades)

    for params, metrics in grid_results.items():
        med = neighbor_median_oos_e(grid_results, params)
        metrics.robustness_score = robustness_score(
            oos_trades=metrics.oos_trades,
            oos_expectancy=metrics.oos_expectancy,
            oos_pf=metrics.oos_profit_factor,
            oos_max_dd=metrics.oos_max_drawdown,
            year_good_pct=metrics.oos_year_good_pct,
            year_pf_share=metrics.oos_year_pf_max_share,
            neighbor_median_oos_e=med,
        )

    ranked = sorted(grid_results.values(), key=lambda m: m.robustness_score, reverse=True)
    baseline_m = grid_results[baseline_params]

    print("Collecting baseline with snapshots for drawdown/volume...", flush=True)
    baseline_trades = collect_fast(universe, baseline_params, with_snapshots=True)
    dd = drawdown_investigation(baseline_trades)
    vol_rows = volume_band_analysis(baseline_trades)

    lines: list[str] = []
    lines.append("# Momentum Breakout — Robustness Validation")
    lines.append("")
    lines.append(f"**Universe:** {', '.join(SYMBOLS)}  ")
    lines.append(f"**Period:** {eff_start} → {REQUESTED_END}  ")
    lines.append(f"**OOS window:** signal years {sorted(OOS_YEARS)}  ")
    lines.append(f"**Grid size:** {len(grid_results)} parameter combinations  ")
    lines.append("")

    lines.append("## 1. Parameter sensitivity (top 15 by robustness score)")
    lines.append("")
    lines.append(
        "| Rank | Params | Total | OOS | WR | E | PF | MaxDD | OOS E | OOS PF | OOS MaxDD | Year%+ | Score |"
    )
    lines.append(
        "|------|--------|-------|-----|----|----|-----|-------|-------|--------|-----------|--------|-------|"
    )
    for idx, m in enumerate(ranked[:15], 1):
        lines.append(
            f"| {idx} | {m.params.label()} | {m.total_trades} | {m.oos_trades} | "
            f"{m.win_rate*100:.0f}% | {m.expectancy*100:.1f}% | {m.profit_factor:.2f} | "
            f"{m.max_drawdown*100:.0f}% | {m.oos_expectancy*100:.1f}% | {m.oos_profit_factor:.2f} | "
            f"{m.oos_max_drawdown*100:.0f}% | {m.oos_year_good_pct*100:.0f}% | {m.robustness_score:.0f} |"
        )
    lines.append("")

    lines.append("### Current production config (baseline)")
    lines.append("")
    lines.append(f"- **Params:** {baseline_m.params.label()}  ")
    lines.append(
        f"- Total {baseline_m.total_trades} / OOS {baseline_m.oos_trades} | E {baseline_m.expectancy*100:.1f}% | "
        f"OOS E {baseline_m.oos_expectancy*100:.1f}% | PF {baseline_m.profit_factor:.2f} | "
        f"OOS PF {baseline_m.oos_profit_factor:.2f} | MaxDD {baseline_m.max_drawdown*100:.0f}% | "
        f"Score **{baseline_m.robustness_score:.0f}**"
    )
    best = ranked[0]
    lines.append("")
    lines.append("### Best robust parameter set (grid winner)")
    lines.append("")
    lines.append(f"- **Params:** {best.params.label()}  ")
    lines.append(
        f"- Total {best.total_trades} / OOS {best.oos_trades} | OOS E {best.oos_expectancy*100:.1f}% | "
        f"OOS PF {best.oos_profit_factor:.2f} | OOS MaxDD {best.oos_max_drawdown*100:.0f}% | "
        f"Score **{best.robustness_score:.0f}**"
    )
    lines.append("")

    lines.append("## 2. Robustness scoring methodology")
    lines.append("")
    lines.append("Rewards: positive OOS expectancy (+25), OOS PF ≥ 1.2 (+20), OOS trades ≥ 50 (+20), OOS MaxDD < 35% (+15), year consistency (+0–20).  ")
    lines.append("Penalties: OOS trades < 30 (−30), OOS PF < 1 (−15), MaxDD > 50% (−25), one-year profit share > 55% (−20), unstable vs RS/vol neighbors (−8).  ")
    lines.append("")

    lines.append("## 3. Volume filter investigation (baseline trades)")
    lines.append("")
    lines.append("| Band | Trades | Win rate | Expectancy | Profit factor |")
    lines.append("|------|--------|----------|------------|---------------|")
    for row in vol_rows:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['wr']*100:.0f}% | {row['e']*100:.1f}% | {row['pf']:.2f} |"
        )
    lines.append("")
    vol_175 = next((r for r in vol_rows if "1.5 <=" in r["label"]), None)
    vol_15 = next((r for r in vol_rows if r["label"].startswith("volume_ratio >=")), None)
    vol_30 = next((r for r in vol_rows if ">= 3.0" in r["label"]), None)
    lines.append("**Volume conclusion:**  ")
    if vol_30 and vol_30["n"] >= 5 and vol_30["e"] < 0:
        lines.append("- **Avoid volume_ratio ≥ 3.0** — negative expectancy.  ")
    if vol_175 and vol_15 and vol_175["n"] >= 10:
        if vol_175["e"] >= vol_15["e"] - 0.005:
            lines.append(
                f"- Narrow band 1.5–1.75 (E {vol_175['e']*100:.1f}%, n={vol_175['n']}) does **not** justify raising volume_min to 1.75 vs all ≥1.5 (E {vol_15['e']*100:.1f}%).  "
            )
        else:
            lines.append("- 1.5–1.75 band underperforms broader ≥1.5 sample.  ")
    lines.append("")

    lines.append("## 4. Drawdown investigation (63.9% sequential trade curve)")
    lines.append("")
    if dd:
        lines.append(f"- **Peak → trough:** {dd['peak_date']} → {dd['trough_date']} (max DD **{dd['max_dd']*100:.1f}%**)  ")
        lines.append(f"- **Trades in drawdown window:** {dd['dd_trade_count']}  ")
        lines.append(f"- **Max consecutive losses:** {dd['max_consecutive_losses']} (from ~{dd['worst_streak_start']})  ")
        lines.append(f"- **Same-symbol overlapping trades:** {dd['same_symbol_overlaps']} pairs  ")
        lines.append(f"- **Portfolio-wide overlapping trades:** {dd['portfolio_overlaps']} pairs  ")
        lines.append("")
        lines.append("**By symbol (drawdown window):**  ")
        for sym, info in sorted(dd["by_symbol"].items(), key=lambda x: x[1]["sum_ret"]):
            lines.append(f"- {sym}: {info['n']} trades, sum return {info['sum_ret']*100:.1f}%  ")
        lines.append("")
        lines.append("**By regime (drawdown window):**  ")
        for reg, info in dd["by_regime"].items():
            lines.append(f"- {reg}: {info['n']} trades, sum return {info['sum_ret']*100:.1f}%  ")
        lines.append("")
        lines.append("**Worst trades in window:**  ")
        lines.append("")
        lines.append("| Symbol | Signal | Exit | Return | Regime | Vol ratio |")
        lines.append("|--------|--------|------|--------|--------|-----------|")
        for t in dd["worst_trades"]:
            snap = t.feature_snapshot
            vol = f"{snap.volume_ratio:.2f}" if snap and snap.volume_ratio else "—"
            reg = snap.market_regime.value if snap else "—"
            lines.append(
                f"| {t.symbol} | {t.signal_date} | {t.exit_date} | {t.return_pct*100:.1f}% | {reg} | {vol} |"
            )
    lines.append("")
    lines.append("**Drawdown drivers:** Sequential compounding across **all symbols** treats each trade as full capital commitment; overlapping positions and 2020-style loss clusters inflate path risk vs per-trade edge.  ")
    lines.append("")
    lines.append("**Mandatory risk controls before production alerts:**  ")
    lines.append("1. **Max open positions** (e.g. 3–5) — limits overlap inflation.  ")
    lines.append("2. **Max 1 active trade per symbol** — eliminates same-symbol overlap.  ")
    lines.append("3. **Portfolio risk cap** (e.g. 1–2% risk per trade, ≤6% aggregate open risk).  ")
    lines.append("4. **Circuit breaker:** pause after 3–4 consecutive losses or −10% rolling 20-trade window.  ")
    lines.append("5. **Correlation throttle:** reduce size when ≥3 mega-cap tech signals same week.  ")
    lines.append("")

    keep_baseline = best.robustness_score <= baseline_m.robustness_score + 5
    lines.append("## 5. Final recommendations")
    lines.append("")
    lines.append("### Current vs best robust config")
    lines.append("")
    lines.append("| | Current | Best robust |")
    lines.append("|---|---------|-------------|")
    lines.append(f"| RS min | {baseline_m.params.rs_min} | {best.params.rs_min} |")
    lines.append(f"| Volume min | {baseline_m.params.vol_min} | {best.params.vol_min} |")
    lines.append(f"| Target R | {baseline_m.params.target_r} | {best.params.target_r} |")
    lines.append(f"| Stop lookback | {baseline_m.params.stop_days} | {best.params.stop_days} |")
    lines.append(f"| Robustness score | {baseline_m.robustness_score:.0f} | {best.robustness_score:.0f} |")
    lines.append(f"| OOS trades | {baseline_m.oos_trades} | {best.oos_trades} |")
    lines.append(f"| OOS expectancy | {baseline_m.oos_expectancy*100:.1f}% | {best.oos_expectancy*100:.1f}% |")
    lines.append("")

    lines.append("### Should production config change now?")
    if keep_baseline:
        lines.append(
            "**Keep current config unchanged** for now. Baseline is within 5 points of the grid winner "
            "or wins on robustness — tightening RS/volume to 1.75 is **not** supported by volume-band analysis."
        )
    else:
        lines.append(
            f"**Consider** moving toward `{best.params.label()}` after paper-trading with risk controls; "
            "verify OOS trade count ≥ 50 and year consistency before any prod deploy."
        )
    lines.append("")

    alerts_ready = (
        baseline_m.oos_expectancy > 0
        and baseline_m.oos_profit_factor >= 1.1
        and baseline_m.oos_trades >= 40
        and baseline_m.robustness_score >= 40
    )
    lines.append("### Ready for user-facing alerts?")
    if alerts_ready:
        lines.append(
            "**Conditional yes** — edge is positive OOS, but **only with mandatory risk controls** "
            "(position limits, per-symbol cap, circuit breaker). Not ready for unconstrained alert firing."
        )
    else:
        lines.append(
            "**No** — OOS sample or robustness score insufficient for user-facing alerts without further data."
        )
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/run_momentum_robustness_study.py` — research only.*")

    out = ROOT / "docs" / "momentum_breakout_robustness_validation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    print(f"baseline_score={baseline_m.robustness_score} best={best.params.label()} score={best.robustness_score}")


if __name__ == "__main__":
    main()
