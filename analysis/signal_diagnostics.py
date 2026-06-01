"""Out-of-sample signal diagnostics for the pattern trend model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from backtest.baselines import build_backtest_analysis
from backtest.metrics import (
    UP_PROB_COLUMN,
    attach_ranking_score,
    compute_information_coefficient,
    compute_rank_ic,
    sharpe_ratio,
)
from backtest.run_backtest import load_labeled_universe
from models.labels import (
    EXCESS_RETURN_COLUMN,
    FUTURE_RETURN_COLUMN,
    get_feature_columns,
    get_label_column,
    get_label_values,
    resolve_label_scheme,
)
from models.pattern_production import production_strategy_config
from models.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    _effective_model_config,
    _probability_columns,
    _slice_window,
    build_model_panel,
    generate_walk_forward_windows,
)
from models.xgb_model import predict_xgb, train_xgb_classifier

TRADES_PER_YEAR = 252 / 5
SCORE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.50, 0.55, "0.50-0.55"),
    (0.55, 0.60, "0.55-0.60"),
    (0.60, 0.65, "0.60-0.65"),
    (0.65, 0.70, "0.65-0.70"),
    (0.70, 0.75, "0.70-0.75"),
    (0.75, float("inf"), "0.75+"),
)
SHAP_SAMPLE_SIZE = 2000


@dataclass(frozen=True)
class WalkForwardArtifacts:
    result: WalkForwardResult
    models: list[Any]
    feature_columns: list[str]


def run_walk_forward_with_models(
    labeled_by_symbol: dict[str, pd.DataFrame],
    config: WalkForwardConfig | None = None,
) -> WalkForwardArtifacts:
    """Run walk-forward validation and retain fitted models per window."""
    cfg = config or WalkForwardConfig()
    scheme = resolve_label_scheme(cfg.label_scheme)
    label_column = get_label_column(scheme)
    model_cfg = _effective_model_config(cfg)
    panel = build_model_panel(labeled_by_symbol)
    feature_cols = get_feature_columns(panel)
    windows = generate_walk_forward_windows(panel["date"], cfg)

    prediction_frames: list[pd.DataFrame] = []
    window_metrics: list[dict[str, Any]] = []
    models: list[Any] = []

    for window in windows:
        train_df = _slice_window(panel, window.train_start, window.train_end)
        test_df = _slice_window(panel, window.test_start, window.test_end)

        if len(train_df) < cfg.min_train_samples or len(test_df) < cfg.min_test_samples:
            continue

        model = train_xgb_classifier(
            train_df[feature_cols],
            train_df[label_column],
            model_cfg,
            label_scheme=scheme,
        )
        models.append(model)
        y_pred, y_proba = predict_xgb(
            model,
            test_df[feature_cols],
            label_scheme=scheme,
            config=model_cfg,
        )

        pred_data: dict[str, Any] = {
            "window_id": window.window_id,
            "symbol": test_df["symbol"].to_numpy(),
            "date": test_df["date"].to_numpy(),
            "y_true": test_df[label_column].to_numpy(),
            "y_pred": y_pred,
            FUTURE_RETURN_COLUMN: test_df[FUTURE_RETURN_COLUMN].to_numpy(),
            EXCESS_RETURN_COLUMN: test_df[EXCESS_RETURN_COLUMN].to_numpy(),
        }
        pred_data.update(
            _probability_columns(get_label_values(scheme), y_proba)
        )
        preds = pd.DataFrame(pred_data)
        prediction_frames.append(preds)
        window_metrics.append(
            {
                "window_id": window.window_id,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "accuracy": float((preds["y_pred"] == preds["y_true"]).mean()),
            }
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    result = WalkForwardResult(
        predictions=predictions,
        window_metrics=window_metrics,
        config=cfg,
    )
    return WalkForwardArtifacts(result=result, models=models, feature_columns=feature_cols)


def collect_oos_predictions(
    symbols: Sequence[str],
    config: WalkForwardConfig,
) -> tuple[pd.DataFrame, list[str], list[Any]]:
    """Load labeled data, run walk-forward, and return OOS predictions."""
    labeled = load_labeled_universe(symbols)
    artifacts = run_walk_forward_with_models(labeled, config=config)
    return artifacts.result.predictions, artifacts.feature_columns, artifacts.models


def aggregate_feature_importance(
    models: Sequence[Any],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Average XGBoost gain-based importances across walk-forward models."""
    if not models:
        return pd.DataFrame(columns=["feature", "importance", "importance_pct"])

    matrix = np.zeros((len(models), len(feature_columns)), dtype="float64")
    for idx, model in enumerate(models):
        matrix[idx] = model.feature_importances_

    mean_importance = matrix.mean(axis=0)
    total = float(mean_importance.sum()) or 1.0
    return (
        pd.DataFrame(
            {
                "feature": list(feature_columns),
                "importance": mean_importance,
                "importance_pct": mean_importance / total,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def compute_shap_summary(
    models: Sequence[Any],
    labeled_by_symbol: dict[str, pd.DataFrame],
    feature_columns: Sequence[str],
    *,
    sample_size: int = SHAP_SAMPLE_SIZE,
    random_state: int = 42,
) -> pd.DataFrame:
    """TreeSHAP-style contributions via XGBoost ``pred_contribs`` on a random OOS sample."""
    if not models:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap", "direction"])

    panel = build_model_panel(labeled_by_symbol)
    feature_cols = list(feature_columns)
    sample = panel[feature_cols].dropna()
    if sample.empty:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap", "direction"])

    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=random_state)

    model = models[-1]
    dmatrix = xgb.DMatrix(sample)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = contribs[:, :-1]
    mean_shap = shap_values.mean(axis=0)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    return (
        pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": mean_abs_shap,
                "mean_shap": mean_shap,
                "direction": np.where(mean_shap >= 0, "positive", "negative"),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def ic_by_group(
    predictions: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    """Compute IC and Rank IC for each group (year, symbol, etc.)."""
    frame = attach_ranking_score(predictions.copy())
    if group_col == "year":
        frame["year"] = pd.to_datetime(frame["date"]).dt.year

    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_col, sort=True):
        rows.append(
            {
                group_col: key,
                "n_predictions": len(group),
                "ic": compute_information_coefficient(group),
                "rank_ic": compute_rank_ic(group),
            }
        )
    return pd.DataFrame(rows)


def ic_by_symbol_timeseries(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pearson/Spearman correlation of score vs excess return within each symbol."""
    frame = attach_ranking_score(predictions.copy())
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        score = group[UP_PROB_COLUMN].astype("float64")
        target = group[EXCESS_RETURN_COLUMN].astype("float64")
        valid = group.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])
        rows.append(
            {
                "symbol": symbol,
                "n_predictions": len(valid),
                "ic": float(score.corr(target)) if len(valid) >= 3 else float("nan"),
                "rank_ic": float(score.corr(target, method="spearman"))
                if len(valid) >= 3
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def score_bucket_analysis(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize realized outcomes by predicted probability bucket."""
    frame = attach_ranking_score(predictions.copy())
    frame = frame.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])
    rows: list[dict[str, Any]] = []

    for low, high, label in SCORE_BUCKETS:
        if np.isinf(high):
            mask = frame[UP_PROB_COLUMN] >= low
        else:
            mask = (frame[UP_PROB_COLUMN] >= low) & (frame[UP_PROB_COLUMN] < high)
        bucket = frame.loc[mask, EXCESS_RETURN_COLUMN].astype("float64")
        if bucket.empty:
            rows.append(
                {
                    "bucket": label,
                    "count": 0,
                    "avg_excess_return": float("nan"),
                    "win_rate": float("nan"),
                    "sharpe": float("nan"),
                }
            )
            continue

        rows.append(
            {
                "bucket": label,
                "count": int(len(bucket)),
                "avg_excess_return": float(bucket.mean()),
                "win_rate": float((bucket > 0).mean()),
                "sharpe": sharpe_ratio(bucket, periods_per_year=TRADES_PER_YEAR),
            }
        )

    return pd.DataFrame(rows)


def _assign_rank_buckets(scores: pd.Series, n_buckets: int) -> pd.Series:
    n = len(scores)
    if n < 2:
        return pd.Series(index=scores.index, dtype="Int64")
    ranks = scores.rank(method="first", ascending=True) - 1.0
    bucket = np.floor(ranks / n * n_buckets).astype("int64")
    return pd.Series(np.clip(bucket, 0, n_buckets - 1), index=scores.index)


def decile_portfolio_analysis(
    predictions: pd.DataFrame,
    *,
    n_buckets: int = 10,
) -> dict[str, Any]:
    """Rank symbols cross-sectionally each date; compare top vs bottom buckets."""
    frame = attach_ranking_score(predictions.copy())
    frame = frame.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])
    frame["date"] = pd.to_datetime(frame["date"])

    bucket_rows: list[dict[str, Any]] = []
    spread_rows: list[dict[str, Any]] = []

    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        group = group.copy()
        group["bucket"] = _assign_rank_buckets(group[UP_PROB_COLUMN], n_buckets=n_buckets)

        top_bucket = int(group["bucket"].max())
        bottom_bucket = int(group["bucket"].min())
        top_excess = float(group.loc[group["bucket"] == top_bucket, EXCESS_RETURN_COLUMN].mean())
        bottom_excess = float(
            group.loc[group["bucket"] == bottom_bucket, EXCESS_RETURN_COLUMN].mean()
        )
        spread_rows.append(
            {
                "date": date,
                "n_symbols": len(group),
                "top_bucket": top_bucket,
                "bottom_bucket": bottom_bucket,
                "top_excess_return": top_excess,
                "bottom_excess_return": bottom_excess,
                "spread": top_excess - bottom_excess,
            }
        )

        for bucket, bucket_group in group.groupby("bucket", sort=True):
            rets = bucket_group[EXCESS_RETURN_COLUMN].astype("float64")
            bucket_rows.append(
                {
                    "date": date,
                    "bucket": int(bucket),
                    "n_symbols": len(bucket_group),
                    "avg_excess_return": float(rets.mean()),
                }
            )

    bucket_df = pd.DataFrame(bucket_rows)
    spread_df = pd.DataFrame(spread_rows)

    bucket_summary = (
        bucket_df.groupby("bucket", sort=True)
        .agg(
            n_observations=("avg_excess_return", "count"),
            avg_excess_return=("avg_excess_return", "mean"),
        )
        .reset_index()
    )
    bucket_summary["bucket_label"] = bucket_summary["bucket"].apply(
        lambda b: f"D{b + 1}" if n_buckets == 10 else f"Q{b + 1}"
    )

    spread_series = spread_df["spread"].astype("float64") if not spread_df.empty else pd.Series(dtype="float64")
    top_series = (
        spread_df["top_excess_return"].astype("float64") if not spread_df.empty else pd.Series(dtype="float64")
    )
    bottom_series = (
        spread_df["bottom_excess_return"].astype("float64")
        if not spread_df.empty
        else pd.Series(dtype="float64")
    )

    effective_buckets = n_buckets
    if not bucket_df.empty:
        avg_symbols = float(bucket_df["n_symbols"].mean())
        effective_buckets = min(n_buckets, max(2, int(round(avg_symbols))))

    return {
        "n_buckets_requested": n_buckets,
        "n_buckets_effective": effective_buckets,
        "bucket_summary": bucket_summary,
        "top_bucket_avg_excess_return": float(top_series.mean()) if not top_series.empty else float("nan"),
        "bottom_bucket_avg_excess_return": float(bottom_series.mean())
        if not bottom_series.empty
        else float("nan"),
        "spread_avg": float(spread_series.mean()) if not spread_series.empty else float("nan"),
        "spread_sharpe": sharpe_ratio(spread_series, periods_per_year=TRADES_PER_YEAR)
        if not spread_series.empty
        else float("nan"),
        "top_bucket_sharpe": sharpe_ratio(top_series, periods_per_year=TRADES_PER_YEAR)
        if not top_series.empty
        else float("nan"),
        "bottom_bucket_sharpe": sharpe_ratio(bottom_series, periods_per_year=TRADES_PER_YEAR)
        if not bottom_series.empty
        else float("nan"),
        "spread_by_date": spread_df,
    }


def compute_daily_ic_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional IC and rank IC series."""
    frame = attach_ranking_score(predictions.copy())
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[UP_PROB_COLUMN, EXCESS_RETURN_COLUMN])

    rows: list[dict[str, Any]] = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 2:
            continue
        score = group[UP_PROB_COLUMN].astype("float64")
        target = group[EXCESS_RETURN_COLUMN].astype("float64")
        if score.nunique(dropna=True) < 2 or target.nunique(dropna=True) < 2:
            continue
        ic = score.corr(target, method="pearson")
        rank_ic = score.corr(target, method="spearman")
        if pd.isna(ic) and pd.isna(rank_ic):
            continue
        rows.append(
            {
                "date": date,
                "n_symbols": len(group),
                "ic": float(ic) if pd.notna(ic) else float("nan"),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def ic_distribution_stats(predictions: pd.DataFrame) -> dict[str, float | int]:
    """Distribution of daily cross-sectional IC values."""
    daily = compute_daily_ic_frame(predictions)
    ic = daily["ic"].dropna()
    rank_ic = daily["rank_ic"].dropna()
    if ic.empty:
        return {
            "ic_mean": float("nan"),
            "ic_median": float("nan"),
            "ic_std": float("nan"),
            "ic_pct_positive": float("nan"),
            "rank_ic_mean": float("nan"),
            "rank_ic_median": float("nan"),
            "rank_ic_std": float("nan"),
            "rank_ic_pct_positive": float("nan"),
            "n_ic_periods": 0,
        }
    return {
        "ic_mean": float(ic.mean()),
        "ic_median": float(ic.median()),
        "ic_std": float(ic.std(ddof=0)),
        "ic_pct_positive": float((ic > 0).mean()),
        "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else float("nan"),
        "rank_ic_median": float(rank_ic.median()) if not rank_ic.empty else float("nan"),
        "rank_ic_std": float(rank_ic.std(ddof=0)) if not rank_ic.empty else float("nan"),
        "rank_ic_pct_positive": float((rank_ic > 0).mean()) if not rank_ic.empty else float("nan"),
        "n_ic_periods": int(len(ic)),
    }


def cross_sectional_breadth_stats(predictions: pd.DataFrame) -> dict[str, Any]:
    """Average symbols per date and observation counts per symbol."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    per_date = frame.groupby("date")["symbol"].nunique()
    per_symbol = frame.groupby("symbol").size().sort_values(ascending=False)
    return {
        "avg_symbols_per_date": float(per_date.mean()) if not per_date.empty else float("nan"),
        "median_symbols_per_date": float(per_date.median()) if not per_date.empty else float("nan"),
        "min_symbols_per_date": int(per_date.min()) if not per_date.empty else 0,
        "max_symbols_per_date": int(per_date.max()) if not per_date.empty else 0,
        "n_rebalance_dates": int(len(per_date)),
        "observations_per_symbol": per_symbol.astype(int).to_dict(),
    }


def quintile_spread_by_year(
    predictions: pd.DataFrame,
    *,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Top-vs-bottom quintile spread statistics for each calendar year."""
    frame = predictions.copy()
    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    rows: list[dict[str, Any]] = []

    for year, group in frame.groupby("year", sort=True):
        summary = decile_portfolio_analysis(group, n_buckets=n_buckets)
        spread = summary["spread_avg"]
        rows.append(
            {
                "year": int(year),
                "n_predictions": len(group),
                "avg_symbols_per_date": float(
                    group.groupby(pd.to_datetime(group["date"]))["symbol"].nunique().mean()
                ),
                "top_quintile_avg_excess_return": summary["top_bucket_avg_excess_return"],
                "bottom_quintile_avg_excess_return": summary["bottom_bucket_avg_excess_return"],
                "spread_avg": spread,
                "spread_positive": bool(spread > 0) if pd.notna(spread) else False,
            }
        )

    return pd.DataFrame(rows)


def run_full_diagnostics(
    symbols: Sequence[str],
    config: WalkForwardConfig,
) -> dict[str, Any]:
    """Run all Phase 1 signal validation diagnostics."""
    labeled = load_labeled_universe(symbols)
    artifacts = run_walk_forward_with_models(labeled, config=config)
    predictions = artifacts.result.predictions
    ranked = attach_ranking_score(predictions)

    importance = aggregate_feature_importance(artifacts.models, artifacts.feature_columns)
    shap_summary = compute_shap_summary(
        artifacts.models,
        labeled,
        artifacts.feature_columns,
    )
    quintile = decile_portfolio_analysis(predictions, n_buckets=5)
    ic_distribution = ic_distribution_stats(predictions)
    breadth = cross_sectional_breadth_stats(predictions)
    quintile_by_year = quintile_spread_by_year(predictions, n_buckets=5)
    backtest = build_backtest_analysis(
        artifacts.result,
        labeled,
        label_scheme=config.label_scheme,
        strategy=production_strategy_config(),
    )

    return {
        "symbols": list(symbols),
        "n_predictions": len(predictions),
        "overall_ic": compute_information_coefficient(ranked),
        "overall_rank_ic": compute_rank_ic(ranked),
        "ic_distribution": ic_distribution,
        "cross_sectional_breadth": breadth,
        "quintile_portfolio": quintile,
        "quintile_spread_by_year": quintile_by_year,
        "strategy_metrics": {
            "sharpe_ratio": backtest["model"]["sharpe_ratio"],
            "profit_factor": backtest["model"]["profit_factor"],
            "max_drawdown": backtest["model"]["max_drawdown"],
            "directional_accuracy": backtest["model"]["directional_accuracy"],
            "n_trades": backtest["model"]["n_trades"],
        },
        "buy_and_hold_metrics": {
            "sharpe_ratio": backtest["buy_and_hold"]["sharpe_ratio"],
            "max_drawdown": backtest["buy_and_hold"]["max_drawdown"],
        },
        "feature_importance": importance,
        "shap_summary": shap_summary,
        "top_positive_predictors": shap_summary[shap_summary["direction"] == "positive"].head(10),
        "top_negative_predictors": shap_summary[shap_summary["direction"] == "negative"].head(10),
        "ic_by_year": ic_by_group(predictions, "year"),
        "ic_by_symbol": ic_by_symbol_timeseries(predictions),
        "score_buckets": score_bucket_analysis(predictions),
        "decile_portfolio": decile_portfolio_analysis(predictions, n_buckets=10),
        "decile_quartile_portfolio": decile_portfolio_analysis(predictions, n_buckets=4),
        "predictions": predictions,
        "walk_forward_result": artifacts.result,
        "labeled_by_symbol": labeled,
    }
