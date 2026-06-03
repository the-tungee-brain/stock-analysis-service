"""Cross-section ranking for one as-of date."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from data.store import load_raw
from ranking_pipeline.config import ModelBackend, RankingPipelineConfig
from ranking_pipeline.regime.detector import compute_spy_regime_series, regime_for_date
from ranking_pipeline.features.parquet_store import load_ranking_features
from ranking_pipeline.features.ranking_features import all_ranking_feature_columns
from ranking_pipeline.ml.dataset import feature_columns_for_ml
from ranking_pipeline.ml.registry import load_models, predict
from ranking_pipeline.scoring.composite import score_universe_slice
from ranking_pipeline.storage.sqlite import RankingStore


def _latest_feature_row(symbol: str, as_of: pd.Timestamp) -> pd.Series | None:
    try:
        df = load_ranking_features(symbol)
    except FileNotFoundError:
        return None
    df = df[df.index <= as_of]
    if df.empty:
        return None
    return df.iloc[-1]


def build_cross_section_panel(
    symbols: list[str],
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    rows: dict[str, pd.Series] = {}
    for symbol in symbols:
        row = _latest_feature_row(symbol, as_of)
        if row is not None:
            rows[symbol] = row
    if not rows:
        return pd.DataFrame()
    panel = pd.DataFrame(rows).T
    panel.index.name = "symbol"
    return panel


def run_ranking(
    symbols: list[str],
    store: RankingStore,
    config: RankingPipelineConfig,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[str, list[dict]]:
    as_of_date = as_of or pd.Timestamp.now("UTC").normalize()
    regime_snapshot = _regime_at(store, config, as_of_date)
    regime_multiplier = regime_snapshot.regime_multiplier if regime_snapshot else 1.0
    regime_id = regime_snapshot.regime_id if regime_snapshot else None

    panel = build_cross_section_panel(symbols, as_of_date)
    if panel.empty:
        raise ValueError("No feature rows available for ranking")

    feature_cols = [c for c in all_ranking_feature_columns() if c in panel.columns]
    score_panel = panel[feature_cols]
    composite_results = score_universe_slice(score_panel, config)
    composite_by_symbol = {r.symbol: r for r in composite_results}

    ml_models = None
    if config.model_backend != ModelBackend.COMPOSITE_ONLY:
        ml_models = load_models(config.artifacts_dir, config.model_backend)

    comp_series = pd.Series({s: r.composite_score for s, r in composite_by_symbol.items()})
    comp_norm = (comp_series - comp_series.mean()) / (comp_series.std(ddof=0) or 1.0)

    ranked_rows: list[dict] = []
    ml_cols = feature_columns_for_ml(panel)
    X_ml = panel.reindex(columns=ml_cols).fillna(0.0)

    for symbol in panel.index:
        comp = composite_by_symbol[symbol]
        ml_prob = None
        expected_excess = None
        if ml_models is not None:
            p, exp = predict(ml_models, X_ml.loc[[symbol]])
            ml_prob = float(p[0])
            expected_excess = float(exp[0])

        if ml_prob is not None:
            final = (
                config.ml_blend_weight * ml_prob
                + config.composite_blend_weight * float(comp_norm.get(symbol, 0.0))
            )
        else:
            final = float(comp.composite_score)

        final *= regime_multiplier

        ranked_rows.append(
            {
                "symbol": symbol,
                "composite_score": comp.composite_score,
                "ml_probability": ml_prob,
                "expected_excess_return": expected_excess,
                "final_score": final,
                "contributions": comp.contributions,
            }
        )

    ranked_rows.sort(key=lambda r: r["final_score"], reverse=True)
    for i, row in enumerate(ranked_rows, start=1):
        row["rank"] = i

    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    store.save_ranking_run(
        run_id,
        as_of_date.strftime("%Y-%m-%d"),
        config.model_backend.value,
        store.active_snapshot_id(),
        ranked_rows,
        regime_id=regime_id,
    )
    store.prune_old_runs(config.keep_runs_days)
    return run_id, ranked_rows


def _regime_at(store: RankingStore, config: RankingPipelineConfig, as_of: pd.Timestamp):
    row = store.get_market_regime(as_of.strftime("%Y-%m-%d"))
    if row:
        from ranking_pipeline.regime.detector import RegimeSnapshot

        meta = json.loads(row.get("metadata_json") or "{}")
        return RegimeSnapshot(
            date=row["date"],
            regime_id=row["regime_id"],
            regime_multiplier=float(row["regime_multiplier"]),
            spy_trend_score=float(meta.get("spy_trend_spread", 0)),
            vol_percentile=float(meta.get("spy_vol_percentile", 0.5)),
            risk_tone=str(meta.get("risk_tone", "neutral")),
        )
    try:
        spy = load_raw(config.benchmark_symbol)
        regime_df = compute_spy_regime_series(spy)
        return regime_for_date(regime_df, as_of)
    except FileNotFoundError:
        return None
