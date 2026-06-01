"""Phase 4 signal durability research audit."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.feature_ablation import run_walk_forward_feature_subset
from analysis.phase2_backtest import run_universe_walk_forward
from analysis.regime_analysis import analyze_regime_performance
from analysis.rolling_diagnostics import build_rolling_diagnostics
from analysis.signal_diagnostics import decile_portfolio_analysis, ic_by_group, ic_by_symbol_timeseries
from backtest.alpha_attribution import build_alpha_attribution_report
from backtest.metrics import compute_information_coefficient, compute_rank_ic, sharpe_ratio
from backtest.production_portfolio import ProductionPortfolioConfig, run_production_portfolio_backtest
from features.feature_groups import (
    feature_group_specs,
    only_group,
    simplicity_feature_columns,
    without_group,
)
from models.labels import EXCESS_RETURN_COLUMN, LABEL_HORIZON_DAYS
from models.pattern_production import production_portfolio_config, production_walk_forward_config
from models.walk_forward import build_model_panel
from models.labels import get_feature_columns

TRADES_PER_YEAR = 252 / LABEL_HORIZON_DAYS


def _portfolio_metrics(predictions: pd.DataFrame, portfolio_cfg: ProductionPortfolioConfig) -> dict[str, float]:
    if predictions.empty or EXCESS_RETURN_COLUMN not in predictions.columns:
        return {"sharpe_ratio": float("nan"), "quintile_spread": float("nan")}

    ranking_cfg = portfolio_cfg.to_ranking_config()
    from backtest.ranking_portfolio import simulate_ranking_portfolio, summarize_portfolio_performance

    period_frame, _ = simulate_ranking_portfolio(predictions, ranking_cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=ranking_cfg.hold_days)
    spread = decile_portfolio_analysis(predictions, n_buckets=5)["spread_avg"]
    return {
        "sharpe_ratio": float(summary["sharpe_ratio"]),
        "quintile_spread": float(spread),
    }


def _prediction_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    if predictions.empty:
        return {"ic": float("nan"), "rank_ic": float("nan"), "n_predictions": 0}
    return {
        "ic": compute_information_coefficient(predictions),
        "rank_ic": compute_rank_ic(predictions),
        "n_predictions": int(len(predictions)),
    }


def run_feature_ablation_audit(
    labeled_by_symbol: dict[str, pd.DataFrame],
    *,
    walk_forward_config=None,
    baseline_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Leave-one-out and only-group ablation across feature families."""
    cfg = walk_forward_config or production_walk_forward_config()
    panel = build_model_panel(labeled_by_symbol)
    all_features = get_feature_columns(panel)
    specs = feature_group_specs(all_features)

    if baseline_predictions is None:
        baseline = run_walk_forward_feature_subset(
            labeled_by_symbol,
            feature_columns=all_features,
            config=cfg,
        )
        baseline_predictions = baseline.result.predictions
    baseline_metrics = _prediction_metrics(baseline_predictions)

    leave_one_out_rows: list[dict[str, Any]] = []
    only_group_rows: list[dict[str, Any]] = []

    for spec in specs:
        without_cols = without_group(all_features, spec.key)
        without_art = run_walk_forward_feature_subset(
            labeled_by_symbol,
            feature_columns=without_cols,
            config=cfg,
        )
        without_metrics = _prediction_metrics(without_art.result.predictions)
        leave_one_out_rows.append(
            {
                "group": spec.key,
                "label": spec.label,
                "n_features": len(spec.columns),
                "ic": without_metrics["ic"],
                "rank_ic": without_metrics["rank_ic"],
                "ic_delta_vs_full": baseline_metrics["ic"] - without_metrics["ic"],
                "rank_ic_delta_vs_full": baseline_metrics["rank_ic"] - without_metrics["rank_ic"],
            }
        )

        only_cols = only_group(all_features, spec.key)
        if not only_cols:
            continue
        only_art = run_walk_forward_feature_subset(
            labeled_by_symbol,
            feature_columns=only_cols,
            config=cfg,
        )
        only_metrics = _prediction_metrics(only_art.result.predictions)
        only_group_rows.append(
            {
                "group": spec.key,
                "label": spec.label,
                "n_features": len(only_cols),
                "ic": only_metrics["ic"],
                "rank_ic": only_metrics["rank_ic"],
                "ic_share_of_full": only_metrics["ic"] / baseline_metrics["ic"]
                if baseline_metrics["ic"] not in (0, float("nan"))
                else float("nan"),
            }
        )

    simple_cols = simplicity_feature_columns(all_features)
    simple_art = run_walk_forward_feature_subset(
        labeled_by_symbol,
        feature_columns=simple_cols,
        config=cfg,
    )
    simple_metrics = _prediction_metrics(simple_art.result.predictions)
    portfolio_cfg = production_portfolio_config()

    return {
        "baseline": {
            "feature_count": len(all_features),
            "features": all_features,
            **baseline_metrics,
            **_portfolio_metrics(baseline_predictions, portfolio_cfg),
        },
        "leave_one_out": pd.DataFrame(leave_one_out_rows).sort_values(
            "ic_delta_vs_full",
            ascending=False,
        ),
        "only_group": pd.DataFrame(only_group_rows).sort_values("ic", ascending=False),
        "simplicity_benchmark": {
            "feature_count": len(simple_cols),
            "features": simple_cols,
            **simple_metrics,
            **_portfolio_metrics(simple_art.result.predictions, portfolio_cfg),
        },
        "baseline_predictions": baseline_predictions,
        "simplicity_predictions": simple_art.result.predictions,
    }


def analyze_symbol_persistence(
    predictions: pd.DataFrame,
    portfolio_result: dict[str, Any],
) -> pd.DataFrame:
    """Per-symbol IC, selection frequency, return and Sharpe contribution."""
    ic_frame = ic_by_symbol_timeseries(predictions)
    concentration = portfolio_result["concentration"]["symbol_exposure"]
    attribution = build_alpha_attribution_report(
        portfolio_result["periods"],
        portfolio_result["panel"],
        predictions,
    )
    returns = attribution["return_contribution"][["symbol", "gross_return_contribution", "contribution_share"]]
    merged = ic_frame.merge(concentration, on="symbol", how="outer").merge(
        returns,
        on="symbol",
        how="outer",
    )

    # Per-symbol Sharpe of quintile membership excess returns.
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    sharpe_rows: list[dict[str, Any]] = []
    ranking_cfg = portfolio_result["ranking_config"]
    top_n = ranking_cfg.top_n

    for symbol, group in frame.groupby("symbol", sort=True):
        rets = group[EXCESS_RETURN_COLUMN].astype("float64").dropna()
        sharpe_rows.append(
            {
                "symbol": symbol,
                "symbol_sharpe": sharpe_ratio(rets, periods_per_year=TRADES_PER_YEAR)
                if len(rets) >= 3
                else float("nan"),
            }
        )
    merged = merged.merge(pd.DataFrame(sharpe_rows), on="symbol", how="left")
    merged = merged.sort_values("gross_return_contribution", ascending=False, na_position="last")
    return merged.reset_index(drop=True)


def classify_symbols(symbol_frame: pd.DataFrame) -> dict[str, list[str]]:
    """Tag persistent winners, losers, and diluters."""
    clean = symbol_frame.dropna(subset=["ic", "selection_rate"], how="any")
    winners = clean[
        (clean["ic"] > 0.03)
        & (clean["gross_return_contribution"].fillna(0) > 0)
    ]["symbol"].astype(str).tolist()
    losers = clean[
        (clean["ic"] < -0.02)
        | (clean["gross_return_contribution"].fillna(0) < -0.01)
    ]["symbol"].astype(str).tolist()
    diluters = clean[
        (clean["selection_rate"] > 0.4)
        & (clean["ic"] < 0)
    ]["symbol"].astype(str).tolist()
    return {
        "persistent_winners": winners,
        "persistent_losers": losers,
        "signal_diluters": diluters,
    }


def analyze_temporal_decay(predictions: pd.DataFrame, period_frame: pd.DataFrame) -> dict[str, Any]:
    """IC/quintile spread by year and rolling windows; estimate decay onset."""
    by_year = ic_by_group(predictions, "year")
    rolling = build_rolling_diagnostics(predictions, period_frame, hold_days=LABEL_HORIZON_DAYS)
    spread = rolling["quintile_spread"]
    rolling_spread = rolling["rolling_quintile_spread"]

    decay_onset = _estimate_decay_onset(rolling["rolling_ic"])
    return {
        "ic_by_year": by_year,
        "rolling_ic": rolling["rolling_ic"],
        "rolling_quintile_spread": rolling_spread,
        "rolling_sharpe": rolling["rolling_sharpe"],
        "signal_trend": rolling["signal_trend"],
        "decay_onset_date": decay_onset,
        "latest_rolling_ic": rolling["latest_rolling_ic"],
        "latest_rolling_quintile_spread": float(rolling_spread.dropna().iloc[-1])
        if not rolling_spread.dropna().empty
        else float("nan"),
    }


def _estimate_decay_onset(rolling_ic: pd.Series) -> pd.Timestamp | None:
    """First date rolling IC fell below zero after previously being positive."""
    clean = rolling_ic.dropna()
    if len(clean) < 20:
        return None
    was_positive = False
    for date, value in clean.items():
        if value > 0.01:
            was_positive = True
        if was_positive and value < 0:
            return pd.Timestamp(date).normalize()
    return None


def run_phase4_audit(
    universe: str = "top20",
    *,
    portfolio_config: ProductionPortfolioConfig | None = None,
) -> dict[str, Any]:
    """Full Phase 4 durability audit for one universe."""
    portfolio_cfg = portfolio_config or production_portfolio_config(universe=universe)
    oos = run_universe_walk_forward(universe, walk_forward_config=production_walk_forward_config())
    labeled = oos["labeled_by_symbol"]
    predictions = oos["predictions"]

    ablation = run_feature_ablation_audit(labeled, baseline_predictions=predictions)
    portfolio = run_production_portfolio_backtest(predictions, labeled, portfolio_cfg)
    symbol_frame = analyze_symbol_persistence(predictions, portfolio)
    symbol_classes = classify_symbols(symbol_frame)
    temporal = analyze_temporal_decay(predictions, portfolio["period_frame"])
    regimes = analyze_regime_performance(predictions)

    return {
        "universe": universe,
        "ablation": ablation,
        "portfolio": portfolio,
        "symbol_persistence": symbol_frame,
        "symbol_classes": symbol_classes,
        "temporal_decay": temporal,
        "regime_attribution": regimes,
        "baseline_ic": _prediction_metrics(predictions),
    }
