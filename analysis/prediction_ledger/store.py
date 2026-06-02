"""Parquet persistence for the prediction ledger."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.paths import LEDGER_DIR, ledger_parquet_path

LEDGER_COLUMNS = [
    "symbol",
    "as_of_date",
    "model_key",
    "model_version",
    "ranking_score",
    "rank",
    "percentile",
    "regime_label",
    "market_regime",
    "vix_regime",
    "expected_outcome",
    "resolved",
    "return_5d",
    "return_spy_5d",
    "excess_return_5d",
    "correct",
    "alpha_captured",
]


def load_ledger(path: Path | None = None) -> pd.DataFrame:
    target = path or ledger_parquet_path()
    if not target.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    frame = pd.read_parquet(target)
    for column in LEDGER_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[LEDGER_COLUMNS].copy()


def save_ledger(frame: pd.DataFrame, path: Path | None = None) -> None:
    target = path or ledger_parquet_path()
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ordered = frame[LEDGER_COLUMNS].copy()
    ordered.to_parquet(target, index=False)


def upsert_rows(rows: list[dict], path: Path | None = None) -> pd.DataFrame:
    if not rows:
        return load_ledger(path)
    incoming = pd.DataFrame(rows)
    existing = load_ledger(path)
    if existing.empty:
        save_ledger(incoming, path)
        return incoming

    keys = ["symbol", "as_of_date", "model_key"]
    merged = pd.concat([existing, incoming], ignore_index=True)
    merged = merged.drop_duplicates(subset=keys, keep="last")
    merged = merged.sort_values(["as_of_date", "symbol"]).reset_index(drop=True)
    save_ledger(merged, path)
    return merged
