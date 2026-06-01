"""CLI runner for walk-forward backtests on stored feature Parquets."""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np
import pandas as pd

from backtest.baselines import build_backtest_analysis
from backtest.config import BacktestStrategyConfig
from backtest.metrics import summarize_predictions
from backtest.symbol_quality import SymbolQualityConfig, filter_recommended_symbols
from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL, ensure_benchmark_ohlcv
from data.symbols import get_symbols, get_universe, list_universe_names
from data.store import load_features
from data.loader import load_symbol
from features.market_context import attach_market_context
from models.labels import LabelScheme, add_labels, get_label_values, resolve_label_scheme
from models.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from models.xgb_model import XGBModelConfig


def load_labeled_universe(symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """Load feature Parquets, attach market context, labels, and SPY excess returns."""
    ensure_benchmark_ohlcv()
    spy_close = load_symbol(BENCHMARK_SYMBOL)["close"]
    vix_close = load_symbol(VIX_SYMBOL)["close"]

    labeled: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        symbol_upper = symbol.strip().upper()
        features = load_features(symbol_upper)
        raw = load_symbol(symbol_upper)
        enriched = attach_market_context(
            features,
            stock_close=raw["close"],
            spy_close=spy_close,
            vix_close=vix_close,
        )
        labeled[symbol_upper] = add_labels(
            enriched,
            raw["close"],
            benchmark_close=spy_close,
        )
    return labeled


def run_backtest(
    symbols: Sequence[str] | None = None,
    config: WalkForwardConfig | None = None,
    *,
    return_labeled: bool = False,
) -> WalkForwardResult | tuple[WalkForwardResult, dict[str, pd.DataFrame]]:
    """Load labeled data and run walk-forward validation."""
    tickers = list(symbols) if symbols else get_symbols()
    labeled = load_labeled_universe(tickers)
    result = run_walk_forward(labeled, config=config)
    if return_labeled:
        return result, labeled
    return result


def _format_float(value: float) -> str:
    if value is None or np.isnan(value):
        return "nan"
    if np.isinf(value):
        return "inf"
    return f"{value:.4f}"


def _format_mean_std(mean: float, std: float) -> str:
    if mean is None or np.isnan(mean):
        return "nan"
    if std is None or np.isnan(std):
        return _format_float(mean)
    return f"{_format_float(mean)} ± {_format_float(std)}"


def _format_trade_stats_block(title: str, stats: dict) -> list[str]:
    return [
        title,
        f"  Trades: {stats.get('n_trades', 0)}",
        f"  Win rate: {_format_float(stats.get('win_rate', float('nan')))}",
        f"  Avg trade return: {_format_float(stats.get('avg_trade_return', float('nan')))}",
        f"  Sharpe ratio: {_format_float(stats.get('sharpe_ratio', float('nan')))}",
        f"  Max drawdown: {_format_float(stats.get('max_drawdown', float('nan')))}",
        f"  Profit factor: {_format_float(stats.get('profit_factor', float('nan')))}",
    ]


def resolve_backtest_symbols(
    *,
    symbols: Sequence[str] | None,
    universe: str | None,
) -> list[str]:
    """Resolve CLI symbol list from ``--symbols`` or ``--universe``."""
    if symbols and universe:
        raise ValueError("Use either --symbols or --universe, not both")
    if universe:
        return get_universe(universe)
    if symbols:
        return [symbol.strip().upper() for symbol in symbols]
    return get_symbols()


def format_recommended_symbols_section(
    recommended: list[dict],
    criteria: SymbolQualityConfig,
) -> list[str]:
    lines = [
        "",
        "Recommended symbols "
        f"(PF >= {criteria.min_pf:.2f}, Sharpe >= {criteria.min_sharpe:.2f}, "
        f"trades >= {criteria.min_trades}):",
    ]
    if not recommended:
        lines.append("  (none)")
        return lines

    for row in recommended:
        lines.append(
            "  - "
            f"{row['symbol']} "
            f"(trades={int(row['n_trades'])}, "
            f"PF={_format_float(row['profit_factor'])}, "
            f"Sharpe={_format_float(row['sharpe_ratio'])})"
        )
    return lines


def format_backtest_report(
    analysis: dict,
    label_scheme: LabelScheme | str,
    *,
    symbol_quality: SymbolQualityConfig | None = None,
) -> str:
    scheme = resolve_label_scheme(label_scheme)
    class_labels = get_label_values(scheme)
    model = analysis["model"]
    buy_hold = analysis["buy_and_hold"]
    random_stats = analysis["random_trades"]
    random_mean = random_stats["mean"]
    random_std = random_stats["std"]
    use_class_weights = analysis.get("use_class_weights", False)
    strategy: BacktestStrategyConfig = analysis["strategy"]
    min_up_prob_text = (
        "none"
        if strategy.min_up_prob is None
        else f"{strategy.min_up_prob:.4f}"
    )

    lines = [
        "Walk-forward backtest summary",
        f"  Label scheme: {scheme.value}",
        f"  Class weights: {'enabled' if use_class_weights else 'disabled'}",
        f"  Min P(up) for trades: {min_up_prob_text}",
        f"  Trade cost (bps): {strategy.trade_cost_bps:.2f}",
        "  Note: buy & hold baseline ignores transaction costs; strategy/random metrics are net of cost.",
        f"  Date range: {buy_hold['start_date'].date()} .. {buy_hold['end_date'].date()}",
        f"  Windows: {model['n_windows']}",
        f"  Predictions: {model['n_predictions']}",
        f"  Directional accuracy: {_format_float(model['directional_accuracy'])}",
        f"  Information coefficient (IC): {_format_float(model.get('information_coefficient', float('nan')))}",
        f"  Rank IC: {_format_float(model.get('rank_ic', float('nan')))}",
    ]

    if scheme in {LabelScheme.BINARY_UPDOWN, LabelScheme.BINARY_OUTPERFORM_SPY}:
        lines.extend(
            [
                f"  Binary accuracy: {_format_float(model.get('binary_accuracy', float('nan')))}",
                f"  Precision (up): {_format_float(model.get('precision_up', float('nan')))}",
                f"  Recall (up): {_format_float(model.get('recall_up', float('nan')))}",
                f"  F1 (up): {_format_float(model.get('f1_up', float('nan')))}",
            ]
        )

    lines.extend(
        [
        "",
        "Model strategy (non-overlapping 5D longs on bullish predictions)",
        *_format_trade_stats_block("", model)[1:],
        ]
    )

    per_class = model.get("per_class_accuracy") or {}
    for label in class_labels:
        value = per_class.get(label, float("nan"))
        lines.append(f"  Class {label} accuracy: {_format_float(value)}")

    lines.extend(
        [
        "",
        "Buy & hold baseline (equal-weight daily)",
        f"  Cumulative return: {_format_float(buy_hold['cumulative_return'])}",
        f"  Sharpe ratio: {_format_float(buy_hold['sharpe_ratio'])}",
        f"  Max drawdown: {_format_float(buy_hold['max_drawdown'])}",
        f"  Volatility: {_format_float(buy_hold['volatility'])}",
        "",
        f"Random-trade baseline ({random_stats['n_runs']} runs, matched trade count)",
        f"  Trades: {random_stats['n_trades']}",
        f"  Win rate: {_format_mean_std(random_mean['win_rate'], random_std['win_rate'])}",
        "  Avg trade return: "
        f"{_format_mean_std(random_mean['avg_trade_return'], random_std['avg_trade_return'])}",
        f"  Sharpe ratio: {_format_mean_std(random_mean['sharpe_ratio'], random_std['sharpe_ratio'])}",
        f"  Max drawdown: {_format_mean_std(random_mean['max_drawdown'], random_std['max_drawdown'])}",
        f"  Profit factor: {_format_mean_std(random_mean['profit_factor'], random_std['profit_factor'])}",
        "",
        "Per-symbol trade stats (net of cost)",
        "  Symbol  Trades  WinRate  AvgRet   Sharpe   MaxDD    PF",
        ]
    )

    for row in analysis.get("per_symbol", []):
        lines.append(
            "  "
            f"{row['symbol']:<6}  "
            f"{int(row['n_trades']):>6}  "
            f"{_format_float(row['win_rate']):>7}  "
            f"{_format_float(row['avg_trade_return']):>7}  "
            f"{_format_float(row['sharpe_ratio']):>7}  "
            f"{_format_float(row['max_drawdown']):>7}  "
            f"{_format_float(row['profit_factor']):>7}"
        )

    quality = symbol_quality or analysis.get("symbol_quality") or SymbolQualityConfig()
    recommended = analysis.get("recommended_symbols")
    if recommended is None:
        recommended = filter_recommended_symbols(analysis.get("per_symbol", []), quality)
    lines.extend(format_recommended_symbols_section(recommended, quality))

    lines.extend(
        [
        "",
        "Per-window trade stats",
        "  W  Test period                      Trades  WinRate  AvgRet   Sharpe   MaxDD",
        ]
    )

    for row in analysis["per_window"]:
        lines.append(
            "  "
            f"{int(row['window_id']):>1}  "
            f"{row['test_start'].date()}..{row['test_end'].date()}  "
            f"{int(row['n_trades']):>6}  "
            f"{_format_float(row['win_rate']):>7}  "
            f"{_format_float(row['avg_trade_return']):>7}  "
            f"{_format_float(row['sharpe_ratio']):>7}  "
            f"{_format_float(row['max_drawdown']):>7}"
        )

    return "\n".join(lines)


def format_compact_backtest_report(analysis: dict) -> str:
    """Short summary: aggregate Sharpe/PF/max DD vs B&H and per-symbol PF/Sharpe."""
    model = analysis["model"]
    buy_hold = analysis["buy_and_hold"]
    lines = [
        "Compact backtest summary",
        "",
        "Aggregate (strategy vs buy-and-hold)",
        f"  {'':12}  {'Sharpe':>8}  {'PF':>8}  {'MaxDD':>8}  {'IC':>8}  {'RankIC':>8}",
        f"  {'Strategy':12}  "
        f"{_format_float(model['sharpe_ratio']):>8}  "
        f"{_format_float(model['profit_factor']):>8}  "
        f"{_format_float(model['max_drawdown']):>8}  "
        f"{_format_float(model.get('information_coefficient', float('nan'))):>8}  "
        f"{_format_float(model.get('rank_ic', float('nan'))):>8}",
        f"  {'Buy & hold':12}  "
        f"{_format_float(buy_hold['sharpe_ratio']):>8}  "
        f"{'n/a':>8}  "
        f"{_format_float(buy_hold['max_drawdown']):>8}  "
        f"{'n/a':>8}  "
        f"{'n/a':>8}",
        "",
        "Per symbol",
        f"  {'Symbol':<8}  {'Sharpe':>8}  {'PF':>8}",
    ]
    for row in analysis.get("per_symbol", []):
        lines.append(
            f"  {row['symbol']:<8}  "
            f"{_format_float(row['sharpe_ratio']):>8}  "
            f"{_format_float(row['profit_factor']):>8}"
        )
    return "\n".join(lines)


def print_compact_backtest_report(
    result: WalkForwardResult,
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    label_scheme: LabelScheme | str | None = None,
    strategy: BacktestStrategyConfig | None = None,
    n_random_runs: int = 30,
    random_state: int = 42,
) -> None:
    scheme = resolve_label_scheme(label_scheme or result.config.label_scheme)
    analysis = build_backtest_analysis(
        result,
        labeled_by_symbol,
        class_labels=get_label_values(scheme),
        label_scheme=scheme,
        strategy=strategy,
        n_random_runs=n_random_runs,
        random_state=random_state,
    )
    print(format_compact_backtest_report(analysis))


def print_backtest_report(
    result: WalkForwardResult,
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    label_scheme: LabelScheme | str | None = None,
    strategy: BacktestStrategyConfig | None = None,
    symbol_quality: SymbolQualityConfig | None = None,
    n_random_runs: int = 30,
    random_state: int = 42,
) -> None:
    scheme = resolve_label_scheme(label_scheme or result.config.label_scheme)
    quality = symbol_quality or SymbolQualityConfig()
    analysis = build_backtest_analysis(
        result,
        labeled_by_symbol,
        class_labels=get_label_values(scheme),
        label_scheme=scheme,
        strategy=strategy,
        n_random_runs=n_random_runs,
        random_state=random_state,
    )
    analysis["symbol_quality"] = quality
    analysis["recommended_symbols"] = filter_recommended_symbols(
        analysis.get("per_symbol", []),
        quality,
    )
    print(format_backtest_report(analysis, scheme, symbol_quality=quality))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run walk-forward backtest on feature Parquets.")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Symbols to include (default: data.symbols.DEFAULT_SYMBOLS)",
    )
    parser.add_argument(
        "--universe",
        default=None,
        help=f"Named symbol universe instead of --symbols (choices: {', '.join(list_universe_names())})",
    )
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--min-train-samples", type=int, default=500)
    parser.add_argument("--min-test-samples", type=int, default=50)
    parser.add_argument(
        "--label-scheme",
        choices=[scheme.value for scheme in LabelScheme],
        default=LabelScheme.ORIGINAL_3CLASS.value,
        help="Target label column to train and evaluate against",
    )
    parser.add_argument(
        "--use-class-weights",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply inverse-frequency (multiclass) or scale_pos_weight (binary) during training",
    )
    parser.add_argument(
        "--random-baseline-runs",
        type=int,
        default=30,
        help="Number of Monte Carlo runs for the random-trade baseline",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random-trade baseline simulations",
    )
    parser.add_argument(
        "--min-up-prob",
        type=float,
        default=None,
        help="Minimum P(up) required to open a non-overlapping long (default: none)",
    )
    parser.add_argument(
        "--trade-cost-bps",
        type=float,
        default=0.0,
        help="Round-trip transaction cost per trade in basis points (default: 0)",
    )
    parser.add_argument(
        "--min-symbol-trades",
        type=int,
        default=50,
        help="Minimum per-symbol trades to qualify as recommended (default: 50)",
    )
    parser.add_argument(
        "--min-symbol-pf",
        type=float,
        default=1.3,
        help="Minimum per-symbol profit factor for recommended symbols (default: 1.3)",
    )
    parser.add_argument(
        "--min-symbol-sharpe",
        type=float,
        default=1.0,
        help="Minimum per-symbol Sharpe for recommended symbols (default: 1.0)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        tickers = resolve_backtest_symbols(symbols=args.symbols, universe=args.universe)
    except ValueError as exc:
        parser.error(str(exc))

    strategy = BacktestStrategyConfig(
        min_up_prob=args.min_up_prob,
        trade_cost_bps=args.trade_cost_bps,
    )
    symbol_quality = SymbolQualityConfig(
        min_trades=args.min_symbol_trades,
        min_pf=args.min_symbol_pf,
        min_sharpe=args.min_symbol_sharpe,
    )

    config = WalkForwardConfig(
        train_years=args.train_years,
        test_years=args.test_years,
        start_date=pd.Timestamp(args.start_date) if args.start_date else None,
        end_date=pd.Timestamp(args.end_date) if args.end_date else None,
        min_train_samples=args.min_train_samples,
        min_test_samples=args.min_test_samples,
        label_scheme=args.label_scheme,
        use_class_weights=args.use_class_weights,
        model_config=XGBModelConfig(),
    )

    result, labeled = run_backtest(tickers, config=config, return_labeled=True)
    print_backtest_report(
        result,
        labeled,
        label_scheme=config.label_scheme,
        strategy=strategy,
        symbol_quality=symbol_quality,
        n_random_runs=args.random_baseline_runs,
        random_state=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
