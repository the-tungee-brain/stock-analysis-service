"""Prediction ledger: record, resolve outcomes, and compute trust stats."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.prediction_ledger.store import load_ledger, upsert_rows
from analysis.research_decision.regime import build_regime_context
from analysis.research_decision.predictions import predict_from_feature_row
from analysis.research_decision.features import build_feature_history
from data.benchmarks import BENCHMARK_SYMBOL
from data.loader import load_symbol
from data.symbols import get_training_universe
from models.labels import LABEL_HORIZON_DAYS
from models.prediction_service import LoadedModel

DEFAULT_LOOKBACK_DAYS = 30


def record_prediction(
    *,
    symbol: str,
    loaded: LoadedModel,
    ranking_row: dict[str, Any] | None,
    regime: dict[str, Any] | None,
    expected_outcome: str | None,
) -> None:
    if ranking_row is None:
        return
    metadata = loaded.metadata
    current = (regime or {}).get("current") or regime or {}
    row = {
        "symbol": symbol.strip().upper(),
        "as_of_date": ranking_row.get("date") or pd.Timestamp.today().strftime("%Y-%m-%d"),
        "model_key": metadata.get("model_key", "model_c"),
        "model_version": metadata.get("train_end_date"),
        "ranking_score": ranking_row.get("ranking_score"),
        "rank": ranking_row.get("rank"),
        "percentile": ranking_row.get("percentile"),
        "regime_label": current.get("regime_label"),
        "market_regime": current.get("market_regime"),
        "vix_regime": current.get("vix_regime"),
        "expected_outcome": expected_outcome,
        "resolved": False,
        "return_5d": np.nan,
        "return_spy_5d": np.nan,
        "excess_return_5d": np.nan,
        "correct": np.nan,
        "alpha_captured": np.nan,
    }
    upsert_rows([row])


def record_universe_today(loaded: LoadedModel, universe_rows: list[dict[str, Any]]) -> None:
    regime = build_regime_context()
    from analysis.research_decision.ranking import ranking_explanation

    rows: list[dict] = []
    metadata = loaded.metadata
    for item in universe_rows:
        rank_block = ranking_explanation(
            rank=item["rank"],
            universe_size=len(universe_rows),
            percentile=item["percentile"],
        )
        rows.append(
            {
                "symbol": item["symbol"],
                "as_of_date": item["date"],
                "model_key": metadata.get("model_key", "model_c"),
                "model_version": metadata.get("train_end_date"),
                "ranking_score": item["ranking_score"],
                "rank": item["rank"],
                "percentile": item["percentile"],
                "regime_label": regime.get("regime_label"),
                "market_regime": regime.get("market_regime"),
                "vix_regime": regime.get("vix_regime"),
                "expected_outcome": rank_block["expected_outcome"],
                "resolved": False,
                "return_5d": np.nan,
                "return_spy_5d": np.nan,
                "excess_return_5d": np.nan,
                "correct": np.nan,
                "alpha_captured": np.nan,
            }
        )
    upsert_rows(rows)


def resolve_outcomes(*, horizon_days: int = LABEL_HORIZON_DAYS) -> pd.DataFrame:
    frame = load_ledger()
    if frame.empty:
        return frame

    spy = load_symbol(BENCHMARK_SYMBOL)["close"].astype("float64")
    updated_rows: list[dict] = []

    for idx, row in frame.iterrows():
        if bool(row.get("resolved")):
            continue
        as_of = pd.Timestamp(row["as_of_date"]).normalize()
        symbol = str(row["symbol"]).upper()
        try:
            closes = load_symbol(symbol)["close"].astype("float64")
        except FileNotFoundError:
            continue

        aligned = closes.index[closes.index.normalize() >= as_of]
        if len(aligned) <= horizon_days:
            continue
        entry_idx = aligned[0]
        exit_idx = aligned[min(horizon_days, len(aligned) - 1)]
        if entry_idx == exit_idx:
            continue

        stock_ret = float(closes.loc[exit_idx] / closes.loc[entry_idx] - 1.0)
        spy_slice = spy.index[spy.index.normalize() >= as_of]
        if len(spy_slice) <= horizon_days:
            continue
        spy_entry = spy_slice[0]
        spy_exit = spy_slice[min(horizon_days, len(spy_slice) - 1)]
        spy_ret = float(spy.loc[spy_exit] / spy.loc[spy_entry] - 1.0)
        excess = stock_ret - spy_ret

        score = float(row["ranking_score"])
        predicted_outperform = score >= 0.5
        actual_outperform = excess > 0
        correct = predicted_outperform == actual_outperform
        alpha = excess if predicted_outperform else -excess

        updated = row.to_dict()
        updated.update(
            {
                "resolved": True,
                "return_5d": stock_ret,
                "return_spy_5d": spy_ret,
                "excess_return_5d": excess,
                "correct": correct,
                "alpha_captured": alpha,
            }
        )
        updated_rows.append(updated)

    if updated_rows:
        upsert_rows(updated_rows)
        frame = load_ledger()
    return frame


def backfill_history(
    loaded: LoadedModel,
    *,
    days: int = DEFAULT_LOOKBACK_DAYS,
    universe: str = "top20",
) -> None:
    symbols = get_training_universe(universe)
    regime = build_regime_context()
    metadata = loaded.metadata
    by_date: dict[str, list[dict]] = {}

    for symbol in symbols:
        try:
            history = build_feature_history(symbol, lookback=days + 5)
        except (FileNotFoundError, ValueError):
            continue
        for as_of in history.index[-days:]:
            row = history.loc[as_of]
            try:
                pred = predict_from_feature_row(row, loaded)
            except Exception:
                continue
            score = pred.get("ranking_score")
            if score is None:
                continue
            date_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
            by_date.setdefault(date_str, []).append(
                {
                    "symbol": symbol.upper(),
                    "ranking_score": float(score),
                }
            )

    rows: list[dict] = []
    for date_str, day_rows in by_date.items():
        day_rows.sort(key=lambda item: item["ranking_score"], reverse=True)
        n = len(day_rows)
        for rank, item in enumerate(day_rows, start=1):
            percentile = int(round(100 * (1 - (rank - 1) / max(n - 1, 1))))
            rows.append(
                {
                    "symbol": item["symbol"],
                    "as_of_date": date_str,
                    "model_key": metadata.get("model_key", "model_c"),
                    "model_version": metadata.get("train_end_date"),
                    "ranking_score": item["ranking_score"],
                    "rank": rank,
                    "percentile": percentile,
                    "regime_label": regime.get("regime_label"),
                    "market_regime": regime.get("market_regime"),
                    "vix_regime": regime.get("vix_regime"),
                    "expected_outcome": None,
                    "resolved": False,
                    "return_5d": np.nan,
                    "return_spy_5d": np.nan,
                    "excess_return_5d": np.nan,
                    "correct": np.nan,
                    "alpha_captured": np.nan,
                }
            )

    if rows:
        upsert_rows(rows)


def ledger_summary(
    *,
    symbol: str | None = None,
    days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    resolve_outcomes()
    frame = load_ledger()
    if frame.empty:
        return _empty_summary(days)

    frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days + LABEL_HORIZON_DAYS)
    view = frame[frame["as_of_date"] >= cutoff]
    if symbol:
        view = view[view["symbol"] == symbol.strip().upper()]

    resolved = view[view["resolved"] == True]  # noqa: E712
    pending = view[view["resolved"] != True]  # noqa: E712

    hit_rate = float(resolved["correct"].mean()) if not resolved.empty else None
    avg_alpha = (
        float(resolved["alpha_captured"].mean()) if not resolved.empty else None
    )

    best = _format_call(resolved, ascending=False)
    worst = _format_call(resolved, ascending=True)

    entries = [
        _entry_dict(row)
        for _, row in view.sort_values("as_of_date", ascending=False).head(days * 3).iterrows()
    ]

    return {
        "days": days,
        "symbol": symbol,
        "n_predictions": int(len(view)),
        "n_resolved": int(len(resolved)),
        "n_pending": int(len(pending)),
        "hit_rate": hit_rate,
        "avg_alpha": avg_alpha,
        "best_call": best,
        "worst_call": worst,
        "entries": entries[:days],
    }


def _entry_dict(row: pd.Series) -> dict[str, Any]:
    return {
        "symbol": row["symbol"],
        "as_of_date": pd.Timestamp(row["as_of_date"]).strftime("%Y-%m-%d"),
        "rank": int(row["rank"]) if pd.notna(row["rank"]) else None,
        "percentile": int(row["percentile"]) if pd.notna(row["percentile"]) else None,
        "ranking_score": float(row["ranking_score"]) if pd.notna(row["ranking_score"]) else None,
        "regime_label": row.get("regime_label"),
        "model_version": row.get("model_version"),
        "expected_outcome": row.get("expected_outcome"),
        "resolved": bool(row.get("resolved")),
        "return_5d": _optional_float(row, "return_5d"),
        "excess_return_5d": _optional_float(row, "excess_return_5d"),
        "correct": bool(row["correct"]) if pd.notna(row.get("correct")) else None,
        "alpha_captured": _optional_float(row, "alpha_captured"),
    }


def _optional_float(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_call(resolved: pd.DataFrame, *, ascending: bool) -> dict[str, Any] | None:
    if resolved.empty:
        return None
    ordered = resolved.sort_values("alpha_captured", ascending=ascending)
    row = ordered.iloc[0]
    return _entry_dict(row)


def _empty_summary(days: int) -> dict[str, Any]:
    return {
        "days": days,
        "symbol": None,
        "n_predictions": 0,
        "n_resolved": 0,
        "n_pending": 0,
        "hit_rate": None,
        "avg_alpha": None,
        "best_call": None,
        "worst_call": None,
        "entries": [],
    }
