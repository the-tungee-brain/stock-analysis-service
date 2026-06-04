from __future__ import annotations

import logging
import math
from typing import Any

from app.services.emerging_leaders_forward_returns import (
    compute_forward_outcomes_for_snapshots,
)
from app.services.emerging_leaders_snapshot_service import (
    capture_emerging_leaders_daily_snapshot,
)
from app.storage.emerging_leaders_validation_store import open_validation_store

logger = logging.getLogger(__name__)

STAGE_LABEL_MAP = {
    "BASE_BUILDING": "Stage 1",
    "TIGHTENING": "Stage 2",
    "BREAKOUT_WATCH": "Stage 3",
}


def backfill_emerging_leaders_forward_returns() -> dict[str, int]:
    store = open_validation_store()
    pending = store.list_snapshots_pending_returns()
    outcomes = compute_forward_outcomes_for_snapshots(pending)
    written = store.upsert_forward_outcomes(outcomes)
    return {"pending": len(pending), "outcomes_written": written}


def run_emerging_leaders_validation_job(
    *,
    snapshot_date: str | None = None,
    force: bool = False,
    skip_snapshot: bool = False,
    skip_backfill: bool = False,
) -> dict[str, Any]:
    """
    Nightly validation: snapshot qualifying setups, then backfill forward returns.
    Does not change live Emerging Leaders ranking.
    """
    result: dict[str, Any] = {}
    if not skip_snapshot:
        snap = capture_emerging_leaders_daily_snapshot(
            snapshot_date=snapshot_date,
            force=force,
        )
        result["snapshot"] = snap
        logger.info("Emerging leaders validation snapshot: %s", snap)
    if not skip_backfill:
        backfill = backfill_emerging_leaders_forward_returns()
        result["backfill"] = backfill
        logger.info("Emerging leaders validation backfill: %s", backfill)
    return result


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _hit_rate(excess_values: list[float | None]) -> float | None:
    clean = [v for v in excess_values if v is not None and not math.isnan(v)]
    if not clean:
        return None
    return float(sum(1 for v in clean if v > 0) / len(clean))


def _bucket_rows(
    rows: list[dict[str, Any]],
    *,
    label_fn,
    bucket_defs: list[tuple[str, callable]],
    horizons: tuple[str, ...] = ("5d", "10d", "20d"),
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for label, predicate in bucket_defs:
        subset = [r for r in rows if predicate(r)]
        bucket: dict[str, Any] = {
            "bucket": label,
            "count": len(subset),
        }
        for h in horizons:
            ret_key = f"ret_{h}"
            excess_key = f"excess_ret_{h}"
            bucket[f"avgRet{h.upper()}"] = _mean(
                [r[ret_key] for r in subset if r.get(ret_key) is not None]
            )
            bucket[f"avgExcess{h.upper()}"] = _mean(
                [r[excess_key] for r in subset if r.get(excess_key) is not None]
            )
            bucket[f"hitRate{h.upper()}"] = _hit_rate(
                [r.get(excess_key) for r in subset]
            )
        results.append(bucket)
    return results


def build_emerging_leaders_validation_dashboard() -> dict[str, Any]:
    store = open_validation_store()
    counts = store.summary_counts()
    rows = store.load_labeled_rows()

    setup_buckets = _bucket_rows(
        rows,
        label_fn=lambda r: r["setup_score"],
        bucket_defs=[
            ("80+", lambda r: r["setup_score"] >= 80),
            ("70–79", lambda r: 70 <= r["setup_score"] <= 79),
            ("60–69", lambda r: 60 <= r["setup_score"] <= 69),
            ("below 60", lambda r: r["setup_score"] < 60),
        ],
    )

    velocity_buckets = _bucket_rows(
        rows,
        label_fn=lambda r: r["compression_velocity"],
        bucket_defs=[
            ("90+", lambda r: r["compression_velocity"] >= 90),
            ("80–89", lambda r: 80 <= r["compression_velocity"] <= 89),
            ("70–79", lambda r: 70 <= r["compression_velocity"] <= 79),
            ("below 70", lambda r: r["compression_velocity"] < 70),
        ],
    )

    stage_buckets = _bucket_rows(
        rows,
        label_fn=lambda r: r["stage"],
        bucket_defs=[
            ("Stage 1", lambda r: r["stage"] == "BASE_BUILDING"),
            ("Stage 2", lambda r: r["stage"] == "TIGHTENING"),
            ("Stage 3", lambda r: r["stage"] == "BREAKOUT_WATCH"),
        ],
    )

    top_decile: dict[str, Any] = {
        "bucket": "top decile (by setup score)",
        "count": 0,
        "avgRet5D": None,
        "avgRet10D": None,
        "avgRet20D": None,
        "avgExcess5D": None,
        "avgExcess10D": None,
        "avgExcess20D": None,
        "hitRate5D": None,
        "hitRate10D": None,
        "hitRate20D": None,
    }
    if rows:
        scores = sorted(r["setup_score"] for r in rows)
        cutoff_idx = max(0, int(math.ceil(len(scores) * 0.9)) - 1)
        cutoff = scores[cutoff_idx]
        decile_rows = [r for r in rows if r["setup_score"] >= cutoff]
        top_decile = _bucket_rows(
            decile_rows,
            label_fn=lambda r: r["setup_score"],
            bucket_defs=[("top decile (by setup score)", lambda r: True)],
        )[0]

    return {
        "snapshotDates": counts["snapshot_dates"],
        "snapshotRows": counts["snapshot_rows"],
        "labeledRows": counts["labeled_rows"],
        "setupScoreBuckets": setup_buckets,
        "compressionVelocityBuckets": velocity_buckets,
        "stageBuckets": stage_buckets,
        "topDecile": top_decile,
        "methodology": (
            "Forward returns use 5/10/20 trading sessions from snapshot date; "
            "excess return vs SPY; universe percentile within same-day cohort."
        ),
    }
