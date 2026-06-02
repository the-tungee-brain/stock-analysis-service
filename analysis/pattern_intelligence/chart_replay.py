"""Forward-return replay validation for detected patterns."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import scan_candlestick_patterns


def build_pattern_replay(
    ohlcv: pd.DataFrame,
    pattern_id: str,
    *,
    current_bar_index: int | None = None,
) -> dict[str, Any] | None:
    if pattern_id not in ohlcv.index and len(ohlcv) < 30:
        return None

    scan = scan_candlestick_patterns(ohlcv)
    hit_col = f"hit_{pattern_id}"
    if hit_col not in scan.columns:
        return None

    hit_mask = scan[hit_col].astype(bool)
    hit_dates = scan.index[hit_mask]
    if len(hit_dates) == 0:
        return None

    returns_5d: list[float] = []
    returns_20d: list[float] = []
    for hit_date in hit_dates:
        loc = ohlcv.index.get_loc(hit_date)
        if isinstance(loc, slice):
            loc = loc.stop - 1
        loc = int(loc)
        if loc + 20 >= len(ohlcv):
            continue
        entry = float(ohlcv["close"].iloc[loc])
        if entry <= 0:
            continue
        returns_5d.append(float(ohlcv["close"].iloc[loc + 5] / entry - 1.0))
        returns_20d.append(float(ohlcv["close"].iloc[loc + 20] / entry - 1.0))

    if not returns_5d:
        return {
            "pattern_id": pattern_id,
            "occurrences": 0,
            "message": "Not enough history to replay forward returns.",
        }

    avg_5 = float(np.mean(returns_5d))
    avg_20 = float(np.mean(returns_20d))
    win_5 = float(sum(1 for value in returns_5d if value > 0) / len(returns_5d))
    win_20 = float(sum(1 for value in returns_20d if value > 0) / len(returns_20d))

    current_note = None
    if current_bar_index is not None and current_bar_index + 20 < len(ohlcv):
        entry = float(ohlcv["close"].iloc[current_bar_index])
        if entry > 0:
            live_5 = float(ohlcv["close"].iloc[current_bar_index + 5] / entry - 1.0)
            live_20 = float(ohlcv["close"].iloc[current_bar_index + 20] / entry - 1.0)
            current_note = (
                f"Most recent signal: {live_5:+.1%} over 5 sessions, "
                f"{live_20:+.1%} over 20 sessions."
            )

    return {
        "pattern_id": pattern_id,
        "occurrences": len(returns_5d),
        "avg_return_5d": round(avg_5, 4),
        "avg_return_20d": round(avg_20, 4),
        "win_rate_5d": round(win_5, 4),
        "win_rate_20d": round(win_20, 4),
        "median_return_5d": round(float(np.median(returns_5d)), 4),
        "median_return_20d": round(float(np.median(returns_20d)), 4),
        "summary": (
            f"Historically after this pattern: avg {avg_5:+.1%} (5d), "
            f"{avg_20:+.1%} (20d) across {len(returns_5d)} occurrences. "
            f"5d win rate {win_5:.0%}."
        ),
        "current_signal_note": current_note,
    }
