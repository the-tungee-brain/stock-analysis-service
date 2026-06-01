"""Out-of-sample performance metrics for walk-forward predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from backtest.config import BacktestStrategyConfig, default_backtest_strategy
from models.labels import FUTURE_RETURN_COLUMN, LabelScheme, get_label_values, resolve_label_scheme, LABEL_HORIZON_DAYS

TRADING_DAYS_PER_YEAR = 252
LONG_SIGNAL = 1
TRADES_PER_YEAR = TRADING_DAYS_PER_YEAR / LABEL_HORIZON_DAYS
UP_PROB_COLUMN = "prob_1"


def compute_directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float((y_true_arr == y_pred_arr).mean())


def compute_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
) -> pd.DataFrame:
    labels = list(class_labels) if class_labels is not None else [-1, 0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_per_class_accuracy(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
) -> dict[int, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    labels = class_labels if class_labels is not None else (-1, 0, 1)
    out: dict[int, float] = {}
    for label in labels:
        mask = y_true_arr == label
        if mask.any():
            out[label] = float((y_pred_arr[mask] == label).mean())
        else:
            out[label] = float("nan")
    return out


def compute_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float | pd.DataFrame]:
    """Precision, recall, F1, and confusion matrix for binary labels ``0/1``."""
    labels = [0, 1]
    cm = compute_confusion_matrix(y_true, y_pred, class_labels=labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "binary_accuracy": compute_directional_accuracy(y_true, y_pred),
        "precision_up": float(precision[1]),
        "recall_up": float(recall[1]),
        "f1_up": float(f1[1]),
        "confusion_matrix": cm,
    }


def strategy_returns(predictions: pd.DataFrame) -> pd.Series:
    """Legacy overlapping simulation: applies ``future_ret_5d`` on every bullish day.

    Prefer ``strategy_returns_non_overlapping()`` for PnL metrics.
    """
    if predictions.empty:
        return pd.Series(dtype="float64")

    ordered = predictions.sort_values(["date", "symbol"]).reset_index(drop=True)
    rets = np.where(
        ordered["y_pred"].to_numpy() == LONG_SIGNAL,
        ordered[FUTURE_RETURN_COLUMN].to_numpy(),
        0.0,
    )
    return pd.Series(rets, dtype="float64")


def apply_trade_cost(return_raw: float, trade_cost_bps: float) -> float:
    """Subtract round-trip cost in basis points from a gross trade return."""
    return float(return_raw) - (trade_cost_bps / 10000.0)


def _resolve_strategy(
    strategy: BacktestStrategyConfig | None,
) -> BacktestStrategyConfig:
    return strategy or default_backtest_strategy()


def _passes_confidence_filter(
    row: Any,
    *,
    min_up_prob: float | None,
    has_prob_column: bool,
) -> bool:
    if min_up_prob is None:
        return True
    if not has_prob_column:
        raise ValueError(
            f"min_up_prob requires {UP_PROB_COLUMN!r} in predictions for confidence filtering"
        )
    return float(getattr(row, UP_PROB_COLUMN)) >= min_up_prob


def simulate_non_overlapping_trades(
    predictions: pd.DataFrame,
    *,
    long_signal: int = LONG_SIGNAL,
    hold_days: int = LABEL_HORIZON_DAYS,
    strategy: BacktestStrategyConfig | None = None,
) -> pd.DataFrame:
    """Build one row per executed 5-day long trade without overlap per symbol."""
    cfg = _resolve_strategy(strategy)
    columns = ["symbol", "entry_date", "exit_date", "return_raw", "return"]
    if predictions.empty:
        return pd.DataFrame(columns=columns)

    required = {"symbol", "date", "y_pred", FUTURE_RETURN_COLUMN}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")

    has_prob_column = UP_PROB_COLUMN in predictions.columns
    trades: list[dict[str, Any]] = []
    ordered = predictions.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])

    for symbol, group in ordered.groupby("symbol", sort=False):
        symbol_rows = group.sort_values("date")
        blocked_until: pd.Timestamp | None = None

        for row in symbol_rows.itertuples(index=False):
            entry_date = pd.Timestamp(row.date)
            if int(row.y_pred) != long_signal:
                continue
            if not _passes_confidence_filter(
                row,
                min_up_prob=cfg.min_up_prob,
                has_prob_column=has_prob_column,
            ):
                continue
            if blocked_until is not None and entry_date <= blocked_until:
                continue

            exit_date = entry_date + pd.offsets.BDay(hold_days)
            return_raw = float(getattr(row, FUTURE_RETURN_COLUMN))
            trades.append(
                {
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "return_raw": return_raw,
                    "return": apply_trade_cost(return_raw, cfg.trade_cost_bps),
                }
            )
            blocked_until = exit_date

    if not trades:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(trades).sort_values(["entry_date", "symbol"]).reset_index(drop=True)


def strategy_returns_non_overlapping(
    predictions: pd.DataFrame,
    *,
    strategy: BacktestStrategyConfig | None = None,
) -> pd.Series:
    """Sequential per-trade net returns from the non-overlapping 5-day long simulation."""
    trades = simulate_non_overlapping_trades(predictions, strategy=strategy)
    if trades.empty:
        return pd.Series(dtype="float64")
    return trades["return"].astype("float64").reset_index(drop=True)


def equity_curve(strategy_rets: pd.Series) -> pd.Series:
    if strategy_rets.empty:
        return pd.Series(dtype="float64")
    return (1.0 + strategy_rets.fillna(0.0)).cumprod()


def sharpe_ratio(
    strategy_rets: pd.Series,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> float:
    if strategy_rets.empty:
        return float("nan")
    rets = strategy_rets.fillna(0.0)
    std = rets.std(ddof=0)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(rets.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def profit_factor(strategy_rets: pd.Series) -> float:
    if strategy_rets.empty:
        return float("nan")
    gains = strategy_rets[strategy_rets > 0].sum()
    losses = -strategy_rets[strategy_rets < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def compute_trade_stats(trade_returns: pd.Series) -> dict[str, float | int]:
    """Summarize executed trade returns."""
    if trade_returns.empty:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "avg_trade_return": float("nan"),
        }

    rets = trade_returns.fillna(0.0)
    return {
        "n_trades": int(len(rets)),
        "win_rate": float((rets > 0).mean()),
        "avg_trade_return": float(rets.mean()),
    }


def summarize_trade_returns(trade_returns: pd.Series) -> dict[str, float | int]:
    """Compute PnL metrics from a sequential trade-return series."""
    trade_stats = compute_trade_stats(trade_returns)
    equity = equity_curve(trade_returns)
    cumulative_return = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    return {
        **trade_stats,
        "cumulative_return": cumulative_return,
        "sharpe_ratio": sharpe_ratio(trade_returns, periods_per_year=TRADES_PER_YEAR),
        "max_drawdown": max_drawdown(equity),
        "profit_factor": profit_factor(trade_returns),
    }


def summarize_daily_returns(daily_returns: pd.Series) -> dict[str, float]:
    """Compute buy-and-hold style metrics from a daily return series."""
    if daily_returns.empty:
        return {
            "cumulative_return": float("nan"),
            "sharpe_ratio": float("nan"),
            "max_drawdown": float("nan"),
            "volatility": float("nan"),
        }

    rets = daily_returns.fillna(0.0)
    equity = equity_curve(rets)
    std = float(rets.std(ddof=0))
    return {
        "cumulative_return": float(equity.iloc[-1] - 1.0),
        "sharpe_ratio": sharpe_ratio(rets, periods_per_year=TRADING_DAYS_PER_YEAR),
        "max_drawdown": max_drawdown(equity),
        "volatility": std * np.sqrt(TRADING_DAYS_PER_YEAR) if std > 0 else float("nan"),
    }


def _empty_classification_summary(
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
    label_scheme: LabelScheme | str | None = None,
) -> dict[str, Any]:
    scheme = resolve_label_scheme(label_scheme) if label_scheme is not None else None
    labels = tuple(class_labels) if class_labels is not None else get_label_values(
        scheme or LabelScheme.ORIGINAL_3CLASS
    )
    summary: dict[str, Any] = {
        "n_predictions": 0,
        "directional_accuracy": float("nan"),
        "per_class_accuracy": {},
        "confusion_matrix": pd.DataFrame(),
        "n_trades": 0,
        "win_rate": float("nan"),
        "avg_trade_return": float("nan"),
        "cumulative_return": float("nan"),
        "sharpe_ratio": float("nan"),
        "max_drawdown": float("nan"),
        "profit_factor": float("nan"),
        "n_windows": 0,
    }
    if scheme == LabelScheme.BINARY_UPDOWN or labels == (0, 1):
        summary.update(
            {
                "binary_accuracy": float("nan"),
                "precision_up": float("nan"),
                "recall_up": float("nan"),
                "f1_up": float("nan"),
            }
        )
    return summary


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    class_labels: tuple[int, ...] | list[int] | None = None,
    label_scheme: LabelScheme | str | None = None,
    strategy: BacktestStrategyConfig | None = None,
) -> dict[str, Any]:
    """Aggregate accuracy, confusion, and non-overlapping trade metrics."""
    cfg = _resolve_strategy(strategy)
    scheme = resolve_label_scheme(label_scheme) if label_scheme is not None else None
    resolved_labels = tuple(class_labels) if class_labels is not None else (
        get_label_values(scheme) if scheme is not None else (-1, 0, 1)
    )

    if predictions.empty:
        return _empty_classification_summary(
            class_labels=resolved_labels,
            label_scheme=scheme,
        )

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    trade_rets = strategy_returns_non_overlapping(predictions, strategy=cfg)
    trade_summary = summarize_trade_returns(trade_rets)

    summary: dict[str, Any] = {
        "n_predictions": int(len(predictions)),
        "directional_accuracy": compute_directional_accuracy(y_true, y_pred),
        "per_class_accuracy": compute_per_class_accuracy(
            y_true,
            y_pred,
            class_labels=resolved_labels,
        ),
        "confusion_matrix": compute_confusion_matrix(
            y_true,
            y_pred,
            class_labels=resolved_labels,
        ),
        **trade_summary,
        "n_windows": int(predictions["window_id"].nunique()),
    }

    is_binary = scheme == LabelScheme.BINARY_UPDOWN or resolved_labels == (0, 1)
    if is_binary:
        binary_metrics = compute_binary_classification_metrics(y_true, y_pred)
        summary["binary_accuracy"] = binary_metrics["binary_accuracy"]
        summary["precision_up"] = binary_metrics["precision_up"]
        summary["recall_up"] = binary_metrics["recall_up"]
        summary["f1_up"] = binary_metrics["f1_up"]
        summary["confusion_matrix"] = binary_metrics["confusion_matrix"]

    return summary


def summarize_per_symbol_trades(
    predictions: pd.DataFrame,
    *,
    strategy: BacktestStrategyConfig | None = None,
) -> list[dict[str, Any]]:
    """Return non-overlapping trade metrics for each symbol in ``predictions``."""
    if predictions.empty:
        return []

    cfg = _resolve_strategy(strategy)
    rows: list[dict[str, Any]] = []
    for symbol in sorted(predictions["symbol"].astype(str).unique()):
        symbol_preds = predictions[predictions["symbol"] == symbol]
        trade_rets = strategy_returns_non_overlapping(symbol_preds, strategy=cfg)
        rows.append({"symbol": symbol, **summarize_trade_returns(trade_rets)})
    return rows
