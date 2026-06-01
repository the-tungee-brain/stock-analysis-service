"""Baseline strategies and per-window reporting for walk-forward backtests."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from backtest.config import BacktestStrategyConfig, default_backtest_strategy
from backtest.metrics import (
    apply_trade_cost,
    simulate_non_overlapping_trades,
    summarize_daily_returns,
    summarize_per_symbol_trades,
    summarize_predictions,
    summarize_trade_returns,
)
from models.labels import FUTURE_RETURN_COLUMN, LABEL_HORIZON_DAYS, LabelScheme
from models.walk_forward import WalkForwardResult

RANDOM_BASELINE_DEFAULT_RUNS = 30
TRADE_METRIC_KEYS: tuple[str, ...] = (
    "win_rate",
    "avg_trade_return",
    "sharpe_ratio",
    "max_drawdown",
    "profit_factor",
)


def get_backtest_date_range(result: WalkForwardResult) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the overall train-start / test-end span for executed windows."""
    if not result.window_metrics:
        raise ValueError("walk-forward result has no executed windows")

    start = min(pd.Timestamp(row["train_start"]) for row in result.window_metrics)
    end = max(pd.Timestamp(row["test_end"]) for row in result.window_metrics)
    return start.normalize(), end.normalize()


def build_equal_weight_daily_returns(
    labeled_by_symbol: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """Equal-weight daily returns across symbols between ``start_date`` and ``end_date``."""
    frames: list[pd.DataFrame] = []
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    for symbol, df in labeled_by_symbol.items():
        piece = df.copy()
        if "date" not in piece.columns:
            piece = piece.reset_index()
            if "date" not in piece.columns:
                piece = piece.rename(columns={piece.columns[0]: "date"})
        piece["date"] = pd.to_datetime(piece["date"]).dt.normalize()
        if "ret_1d" not in piece.columns:
            raise ValueError(f"labeled data for {symbol} is missing ret_1d")

        mask = (piece["date"] >= start) & (piece["date"] <= end)
        subset = piece.loc[mask, ["date", "ret_1d"]].copy()
        subset["symbol"] = symbol.strip().upper()
        frames.append(subset)

    if not frames:
        return pd.Series(dtype="float64")

    panel = pd.concat(frames, ignore_index=True)
    return panel.groupby("date", sort=True)["ret_1d"].mean().sort_index()


def compute_buy_and_hold_baseline(
    labeled_by_symbol: dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> dict[str, Any]:
    """Equal-weight buy-and-hold metrics over the requested date span."""
    daily_returns = build_equal_weight_daily_returns(
        labeled_by_symbol,
        start_date,
        end_date,
    )
    stats = summarize_daily_returns(daily_returns)
    return {
        "start_date": pd.Timestamp(start_date).normalize(),
        "end_date": pd.Timestamp(end_date).normalize(),
        **stats,
    }


def _trade_exit_date(entry_date: pd.Timestamp, hold_days: int = LABEL_HORIZON_DAYS) -> pd.Timestamp:
    return pd.Timestamp(entry_date) + pd.offsets.BDay(hold_days)


def _trades_overlap(
    entry_a: pd.Timestamp,
    entry_b: pd.Timestamp,
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
) -> bool:
    entry_a = pd.Timestamp(entry_a)
    entry_b = pd.Timestamp(entry_b)
    return entry_a <= _trade_exit_date(entry_b, hold_days) and entry_b <= _trade_exit_date(
        entry_a,
        hold_days,
    )


def count_max_non_overlapping_trades(
    dates: Sequence[pd.Timestamp],
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
) -> int:
    """Return the maximum number of non-overlapping entries from ``dates``."""
    return len(build_max_non_overlapping_trade_set(
        dates,
        [0.0] * len(dates),
        hold_days=hold_days,
    ))


def build_max_non_overlapping_trade_set(
    dates: Sequence[pd.Timestamp],
    returns: Sequence[float],
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
) -> list[tuple[pd.Timestamp, float]]:
    """Return the largest non-overlapping trade set in chronological order."""
    selected: list[tuple[pd.Timestamp, float]] = []
    blocked_until: pd.Timestamp | None = None

    for entry_date, trade_return in sorted(
        zip(dates, returns, strict=True),
        key=lambda item: pd.Timestamp(item[0]),
    ):
        entry_date = pd.Timestamp(entry_date)
        if blocked_until is not None and entry_date <= blocked_until:
            continue
        selected.append((entry_date, float(trade_return)))
        blocked_until = _trade_exit_date(entry_date, hold_days)

    return selected


def select_random_non_overlapping_trades(
    dates: Sequence[pd.Timestamp],
    returns: Sequence[float],
    n_trades: int,
    rng: np.random.Generator,
    *,
    hold_days: int = LABEL_HORIZON_DAYS,
) -> list[tuple[pd.Timestamp, float]]:
    """Pick ``n_trades`` random non-overlapping entries from candidate windows."""
    if n_trades <= 0:
        return []

    max_set = build_max_non_overlapping_trade_set(dates, returns, hold_days=hold_days)
    if not max_set:
        return []

    if n_trades > len(max_set):
        raise ValueError(
            f"Requested {n_trades} non-overlapping trades but only {len(max_set)} "
            f"can fit in {len(dates)} candidate days"
        )

    if n_trades == len(max_set):
        return max_set

    chosen_idx = rng.choice(len(max_set), size=n_trades, replace=False)
    return [max_set[int(idx)] for idx in sorted(chosen_idx)]


def simulate_random_non_overlapping_trades(
    predictions: pd.DataFrame,
    trade_counts_by_symbol: dict[str, int],
    rng: np.random.Generator,
    *,
    trade_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Build one random non-overlapping trade set matching per-symbol trade counts."""
    columns = ["symbol", "entry_date", "exit_date", "return_raw", "return"]
    if predictions.empty or not trade_counts_by_symbol:
        return pd.DataFrame(columns=columns)

    candidates = predictions[["symbol", "date", FUTURE_RETURN_COLUMN]].copy()
    candidates["date"] = pd.to_datetime(candidates["date"])
    trades: list[dict[str, Any]] = []

    for symbol, group in candidates.groupby("symbol", sort=False):
        n_trades = int(trade_counts_by_symbol.get(symbol, 0))
        if n_trades <= 0:
            continue

        group = group.sort_values("date")
        selected = select_random_non_overlapping_trades(
            group["date"].tolist(),
            group[FUTURE_RETURN_COLUMN].tolist(),
            n_trades,
            rng,
        )
        for entry_date, trade_return in selected:
            return_raw = float(trade_return)
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": entry_date + pd.offsets.BDay(LABEL_HORIZON_DAYS),
                    "return_raw": return_raw,
                    "return": apply_trade_cost(return_raw, trade_cost_bps),
                }
            )

    if not trades:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(trades).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def compute_random_trade_baseline(
    predictions: pd.DataFrame,
    model_trades: pd.DataFrame,
    *,
    n_runs: int = RANDOM_BASELINE_DEFAULT_RUNS,
    random_state: int = 42,
    trade_cost_bps: float = 0.0,
) -> dict[str, Any]:
    """Monte Carlo random non-overlapping trades matched to model trade counts."""
    if predictions.empty:
        empty_stats = {key: float("nan") for key in TRADE_METRIC_KEYS}
        return {
            "n_runs": n_runs,
            "n_trades": 0,
            "mean": empty_stats,
            "std": empty_stats,
        }

    trade_counts = model_trades.groupby("symbol").size().astype(int).to_dict()
    total_trades = int(model_trades.shape[0]) if not model_trades.empty else 0
    rng = np.random.default_rng(random_state)
    run_summaries: list[dict[str, float | int]] = []

    for _ in range(n_runs):
        random_trades = simulate_random_non_overlapping_trades(
            predictions,
            trade_counts,
            rng,
            trade_cost_bps=trade_cost_bps,
        )
        trade_returns = random_trades["return"].astype("float64").reset_index(drop=True)
        run_summaries.append(summarize_trade_returns(trade_returns))

    return {
        "n_runs": n_runs,
        "n_trades": total_trades,
        "mean": _aggregate_run_metric_stats(run_summaries, "mean"),
        "std": _aggregate_run_metric_stats(run_summaries, "std"),
    }


def summarize_per_window(
    result: WalkForwardResult,
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
    label_scheme: LabelScheme | str | None = None,
    strategy: BacktestStrategyConfig | None = None,
) -> list[dict[str, Any]]:
    """Compute classification and trade metrics for each executed walk-forward window."""
    scheme = label_scheme or result.config.label_scheme
    rows: list[dict[str, Any]] = []

    for window in result.window_metrics:
        window_id = int(window["window_id"])
        window_preds = result.predictions[result.predictions["window_id"] == window_id]
        summary = summarize_predictions(
            window_preds,
            class_labels=class_labels,
            label_scheme=scheme,
            strategy=strategy,
        )
        rows.append(
            {
                "window_id": window_id,
                "train_start": pd.Timestamp(window["train_start"]),
                "train_end": pd.Timestamp(window["train_end"]),
                "test_start": pd.Timestamp(window["test_start"]),
                "test_end": pd.Timestamp(window["test_end"]),
                "n_predictions": summary["n_predictions"],
                "directional_accuracy": summary["directional_accuracy"],
                "n_trades": summary["n_trades"],
                "win_rate": summary["win_rate"],
                "avg_trade_return": summary["avg_trade_return"],
                "sharpe_ratio": summary["sharpe_ratio"],
                "max_drawdown": summary["max_drawdown"],
            }
        )

    return rows


def build_backtest_analysis(
    result: WalkForwardResult,
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
    label_scheme: LabelScheme | str | None = None,
    strategy: BacktestStrategyConfig | None = None,
    n_random_runs: int = RANDOM_BASELINE_DEFAULT_RUNS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Aggregate model, baseline, and per-window stats for reporting."""
    cfg = strategy or default_backtest_strategy()
    scheme = label_scheme or result.config.label_scheme
    scheme_labels = class_labels
    model_summary = summarize_predictions(
        result.predictions,
        class_labels=scheme_labels,
        label_scheme=scheme,
        strategy=cfg,
    )
    model_trades = simulate_non_overlapping_trades(result.predictions, strategy=cfg)
    start_date, end_date = get_backtest_date_range(result)

    return {
        "model": model_summary,
        "label_scheme": scheme,
        "use_class_weights": bool(result.config.use_class_weights),
        "strategy": cfg,
        "buy_and_hold": compute_buy_and_hold_baseline(
            labeled_by_symbol,
            start_date,
            end_date,
        ),
        "random_trades": compute_random_trade_baseline(
            result.predictions,
            model_trades,
            n_runs=n_random_runs,
            random_state=random_state,
            trade_cost_bps=cfg.trade_cost_bps,
        ),
        "per_window": summarize_per_window(
            result,
            class_labels=scheme_labels,
            label_scheme=scheme,
            strategy=cfg,
        ),
        "per_symbol": summarize_per_symbol_trades(result.predictions, strategy=cfg),
        "date_range": {"start_date": start_date, "end_date": end_date},
    }


def _aggregate_run_metric_stats(
    run_summaries: Sequence[dict[str, float | int]],
    reducer: str,
) -> dict[str, float]:
    if not run_summaries:
        return {key: float("nan") for key in TRADE_METRIC_KEYS}

    out: dict[str, float] = {}
    for key in TRADE_METRIC_KEYS:
        values = pd.Series([row[key] for row in run_summaries], dtype="float64")
        if reducer == "mean":
            out[key] = float(values.mean())
        else:
            out[key] = float(values.std(ddof=0))
    return out
