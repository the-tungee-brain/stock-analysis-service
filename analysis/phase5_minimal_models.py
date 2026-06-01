"""Phase 5 minimum viable alpha model comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.feature_ablation import run_walk_forward_feature_subset
from analysis.phase2_backtest import run_universe_walk_forward
from analysis.regime_analysis import build_market_regime_frame
from analysis.signal_diagnostics import compute_daily_ic_frame, decile_portfolio_analysis, ic_by_group
from backtest.metrics import compute_information_coefficient, compute_rank_ic, sharpe_ratio
from backtest.production_portfolio import ProductionPortfolioConfig, run_production_portfolio_backtest
from backtest.ranking_portfolio import simulate_ranking_portfolio, summarize_portfolio_performance
from features.feature_groups import PHASE5_MODELS, phase5_model_features, phase5_model_label
from models.labels import EXCESS_RETURN_COLUMN, LABEL_HORIZON_DAYS
from models.pattern_production import production_portfolio_config, production_walk_forward_config
from models.walk_forward import build_model_panel
from models.labels import get_feature_columns

TRADES_PER_YEAR = 252 / LABEL_HORIZON_DAYS


def _evaluate_model(
    predictions: pd.DataFrame,
    portfolio_cfg: ProductionPortfolioConfig,
    *,
    regime_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ranking_cfg = portfolio_cfg.to_ranking_config()
    period_frame, periods = simulate_ranking_portfolio(predictions, ranking_cfg)
    summary = summarize_portfolio_performance(period_frame, hold_days=ranking_cfg.hold_days)
    spread = decile_portfolio_analysis(predictions, n_buckets=5)

    daily_ic = compute_daily_ic_frame(predictions)
    ic_by_year = ic_by_group(predictions, "year") if not predictions.empty else pd.DataFrame()

    stability = _stability_metrics(daily_ic, ic_by_year)
    regimes = regime_frame if regime_frame is not None else build_market_regime_frame()
    regime_sharpe = _regime_sharpe(period_frame, regimes, hold_days=ranking_cfg.hold_days)

    return {
        "ic": compute_information_coefficient(predictions),
        "rank_ic": compute_rank_ic(predictions),
        "sharpe_ratio": float(summary["sharpe_ratio"]),
        "sortino_ratio": float(summary["sortino_ratio"]),
        "quintile_spread": float(spread["spread_avg"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "avg_turnover": float(summary["avg_turnover"]),
        "n_predictions": int(len(predictions)),
        "stability": stability,
        "regime_sharpe": regime_sharpe,
        "period_frame": period_frame,
    }


def _stability_metrics(daily_ic: pd.DataFrame, ic_by_year: pd.DataFrame) -> dict[str, float]:
    if daily_ic.empty:
        return {
            "ic_std": float("nan"),
            "positive_ic_pct": float("nan"),
            "positive_years_pct": float("nan"),
        }
    ic_series = daily_ic["ic"].astype("float64")
    year_ic = ic_by_year["ic"].astype("float64") if not ic_by_year.empty else pd.Series(dtype="float64")
    return {
        "ic_std": float(ic_series.std(ddof=0)),
        "positive_ic_pct": float((ic_series > 0).mean()),
        "positive_years_pct": float((year_ic > 0).mean()) if not year_ic.empty else float("nan"),
    }


def _regime_sharpe(
    period_frame: pd.DataFrame,
    regime_frame: pd.DataFrame,
    *,
    hold_days: int,
) -> dict[str, float]:
    if period_frame.empty:
        return {
            "bull_sharpe": float("nan"),
            "bear_sharpe": float("nan"),
            "high_vix_sharpe": float("nan"),
            "medium_vix_sharpe": float("nan"),
            "low_vix_sharpe": float("nan"),
        }

    periods_per_year = TRADES_PER_YEAR
    frame = period_frame.copy()
    frame["entry_date"] = pd.to_datetime(frame["entry_date"]).dt.normalize()
    regimes = regime_frame.reset_index(names="date")
    merged = frame.merge(
        regimes[["date", "market_regime", "vix_regime"]],
        left_on="entry_date",
        right_on="date",
        how="left",
    )
    rets = merged["net_return"].astype("float64")

    def _sharpe_for_mask(mask: pd.Series) -> float:
        subset = rets[mask.fillna(False)]
        return sharpe_ratio(subset, periods_per_year=periods_per_year) if len(subset) >= 3 else float("nan")

    return {
        "bull_sharpe": _sharpe_for_mask(merged["market_regime"] == "bull"),
        "bear_sharpe": _sharpe_for_mask(merged["market_regime"] == "bear"),
        "high_vix_sharpe": _sharpe_for_mask(merged["vix_regime"] == "high"),
        "medium_vix_sharpe": _sharpe_for_mask(merged["vix_regime"] == "medium"),
        "low_vix_sharpe": _sharpe_for_mask(merged["vix_regime"] == "low"),
    }


def _simplicity_row(model_key: str, n_features: int, metrics: dict[str, Any]) -> dict[str, Any]:
    performance = float(metrics["sharpe_ratio"])
    per_feature = performance / n_features if n_features > 0 else float("nan")
    return {
        "model": model_key,
        "label": phase5_model_label(model_key),
        "n_features": n_features,
        "performance_sharpe": performance,
        "performance_per_feature": per_feature,
        "ic": metrics["ic"],
        "quintile_spread": metrics["quintile_spread"],
    }


def run_phase5_comparison(
    universe: str = "top20",
    *,
    portfolio_config: ProductionPortfolioConfig | None = None,
    reuse_full_model_predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run walk-forward for models A–F and build complexity frontier tables."""
    portfolio_cfg = portfolio_config or production_portfolio_config(universe=universe)
    wf_cfg = production_walk_forward_config()

    if reuse_full_model_predictions is not None:
        oos = run_universe_walk_forward(universe, walk_forward_config=wf_cfg)
        labeled = oos["labeled_by_symbol"]
        full_predictions = reuse_full_model_predictions
    else:
        oos = run_universe_walk_forward(universe, walk_forward_config=wf_cfg)
        labeled = oos["labeled_by_symbol"]
        full_predictions = oos["predictions"]

    panel = build_model_panel(labeled)
    all_features = get_feature_columns(panel)
    regime_frame = build_market_regime_frame()

    model_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    simplicity_rows: list[dict[str, Any]] = []

    for model_key, _, _ in PHASE5_MODELS:
        features = phase5_model_features(all_features, model_key)
        if model_key == "F":
            predictions = full_predictions
        else:
            artifacts = run_walk_forward_feature_subset(
                labeled,
                feature_columns=features,
                config=wf_cfg,
            )
            predictions = artifacts.result.predictions

        metrics = _evaluate_model(predictions, portfolio_cfg, regime_frame=regime_frame)
        model_results[model_key] = {
            "label": phase5_model_label(model_key),
            "features": features,
            "n_features": len(features),
            "predictions": predictions,
            **metrics,
        }

        summary_rows.append(
            {
                "model": model_key,
                "label": phase5_model_label(model_key),
                "n_features": len(features),
                "ic": metrics["ic"],
                "rank_ic": metrics["rank_ic"],
                "sharpe": metrics["sharpe_ratio"],
                "sortino": metrics["sortino_ratio"],
                "quintile_spread": metrics["quintile_spread"],
                "max_drawdown": metrics["max_drawdown"],
                "turnover": metrics["avg_turnover"],
            }
        )
        stability_rows.append({"model": model_key, **metrics["stability"]})
        regime_rows.append({"model": model_key, **metrics["regime_sharpe"]})
        simplicity_rows.append(_simplicity_row(model_key, len(features), metrics))

    summary = pd.DataFrame(summary_rows)
    full_sharpe = float(summary.loc[summary["model"] == "F", "sharpe"].iloc[0])
    full_ic = float(summary.loc[summary["model"] == "F", "ic"].iloc[0])

    summary["sharpe_vs_full"] = summary["sharpe"] / full_sharpe if full_sharpe else float("nan")
    summary["ic_vs_full"] = summary["ic"] / full_ic if full_ic else float("nan")

    simplicity = pd.DataFrame(simplicity_rows).sort_values("n_features")
    frontier = summary.sort_values("n_features").reset_index(drop=True)

    return {
        "universe": universe,
        "models": model_results,
        "summary": summary,
        "stability": pd.DataFrame(stability_rows),
        "regime_sharpe": pd.DataFrame(regime_rows),
        "simplicity": simplicity,
        "frontier": frontier,
        "full_model_sharpe": full_sharpe,
        "full_model_ic": full_ic,
    }
