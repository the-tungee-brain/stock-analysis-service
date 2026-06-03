import numpy as np
import pandas as pd
import pytest

from app.services.emerging_leaders_forward_returns import (
    _forward_simple_return,
    compute_forward_outcomes_for_snapshots,
)
from app.services.emerging_leaders_validation_service import (
    _bucket_rows,
    _hit_rate,
    _mean,
)
from app.storage.emerging_leaders_validation_store import EmergingLeadersValidationStore


def _make_calendar_closes(rows: int = 80, start: float = 100.0) -> pd.Series:
    idx = pd.bdate_range("2024-01-02", periods=rows)
    close = start + np.arange(rows) * 0.1
    return pd.Series(close, index=idx)


def test_forward_simple_return_positive():
    calendar = _make_calendar_closes().index
    close = _make_calendar_closes()
    entry = calendar[10]
    ret = _forward_simple_return(close, entry, 5, calendar)
    assert ret is not None
    assert ret > 0


def test_bucket_metrics_and_hit_rate():
    rows = [
        {"setup_score": 85, "compression_velocity": 92, "stage": "TIGHTENING",
         "ret_5d": 0.04, "excess_ret_5d": 0.02, "ret_10d": 0.05, "excess_ret_10d": 0.03,
         "ret_20d": 0.06, "excess_ret_20d": 0.04},
        {"setup_score": 55, "compression_velocity": 45, "stage": "BASE_BUILDING",
         "ret_5d": -0.01, "excess_ret_5d": -0.02, "ret_10d": -0.02, "excess_ret_10d": -0.03,
         "ret_20d": -0.03, "excess_ret_20d": -0.04},
    ]
    buckets = _bucket_rows(
        rows,
        label_fn=lambda r: r["setup_score"],
        bucket_defs=[
            ("80+", lambda r: r["setup_score"] >= 80),
            ("below 60", lambda r: r["setup_score"] < 60),
        ],
    )
    assert buckets[0]["count"] == 1
    assert buckets[0]["avgRet5D"] == 0.04
    assert buckets[0]["hitRate5D"] == 1.0
    assert buckets[1]["hitRate5D"] == 0.0
    assert abs(_mean([0.1, 0.2]) - 0.15) < 1e-9
    assert _hit_rate([0.01, -0.02, 0.03]) == 2 / 3


def test_validation_store_roundtrip(tmp_path):
    db = tmp_path / "el_validation.db"
    store = EmergingLeadersValidationStore(db)
    written = store.insert_snapshot_rows(
        "2024-06-01",
        [
            {
                "symbol": "AAA",
                "rank": 1,
                "setup_score": 78,
                "compression_velocity": 81,
                "setup_purity": 55.0,
                "stage": "TIGHTENING",
            }
        ],
    )
    assert written == 1
    assert store.has_snapshot_date("2024-06-01")
    store.upsert_forward_outcomes(
        [
            {
                "snapshot_id": 1,
                "ret_5d": 0.02,
                "ret_10d": 0.03,
                "ret_20d": 0.04,
                "spy_ret_5d": 0.01,
                "spy_ret_10d": 0.015,
                "spy_ret_20d": 0.02,
                "excess_ret_5d": 0.01,
                "excess_ret_10d": 0.015,
                "excess_ret_20d": 0.02,
                "universe_pct_rank_5d": 70.0,
                "universe_pct_rank_10d": 65.0,
                "universe_pct_rank_20d": 60.0,
            }
        ]
    )
    labeled = store.load_labeled_rows()
    assert len(labeled) == 1
    assert labeled[0]["symbol"] == "AAA"


def test_compute_forward_outcomes_for_snapshots(monkeypatch):
    calendar = pd.bdate_range("2024-01-02", periods=60)
    spy_close = pd.Series(100 + np.arange(60) * 0.2, index=calendar)

    def fake_load(sym):
        if sym == "SPY":
            return pd.DataFrame({"close": spy_close})
        return pd.DataFrame({"close": 50 + np.arange(60) * 0.15}, index=calendar)

    monkeypatch.setattr(
        "app.services.emerging_leaders_forward_returns.load_raw",
        fake_load,
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_forward_returns.raw_exists",
        lambda _s: True,
    )
    monkeypatch.setattr(
        "app.services.emerging_leaders_forward_returns.ensure_benchmark_ohlcv",
        lambda: None,
    )

    pending = [
        {
            "id": 1,
            "snapshot_date": "2024-01-15",
            "symbol": "AAA",
            "rank": 1,
            "setup_score": 70,
            "compression_velocity": 75,
            "setup_purity": 50.0,
            "stage": "TIGHTENING",
        }
    ]
    outcomes = compute_forward_outcomes_for_snapshots(pending)
    assert len(outcomes) == 1
    assert outcomes[0]["ret_5d"] is not None
