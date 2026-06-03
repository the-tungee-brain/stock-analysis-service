from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from data.benchmarks import ensure_benchmark_ohlcv
from data.store import load_raw, raw_exists


def _trading_calendar() -> pd.DatetimeIndex:
    ensure_benchmark_ohlcv()
    spy = load_raw("SPY")
    if spy is None or spy.empty:
        raise LookupError("SPY OHLCV required for trading calendar")
    idx = pd.DatetimeIndex(pd.to_datetime(spy.index)).sort_values().normalize()
    return idx.unique()


def _nearest_calendar_pos(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> int | None:
    date = pd.Timestamp(date).normalize()
    pos = calendar.searchsorted(date, side="right") - 1
    if pos < 0:
        return None
    if calendar[pos] > date:
        pos -= 1
    if pos < 0:
        return None
    return int(pos)


def _forward_simple_return(
    close: pd.Series,
    entry_date: pd.Timestamp,
    horizon_sessions: int,
    calendar: pd.DatetimeIndex,
) -> float | None:
    entry_pos = _nearest_calendar_pos(calendar, entry_date)
    if entry_pos is None:
        return None
    exit_pos = entry_pos + horizon_sessions
    if exit_pos >= len(calendar):
        return None
    entry_ts = calendar[entry_pos]
    exit_ts = calendar[exit_pos]
    series = close.copy()
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).normalize()
    if entry_ts not in series.index or exit_ts not in series.index:
        entry_px = series.asof(entry_ts)
        exit_px = series.asof(exit_ts)
    else:
        entry_px = series.loc[entry_ts]
        exit_px = series.loc[exit_ts]
    if entry_px is None or exit_px is None or pd.isna(entry_px) or pd.isna(exit_px):
        return None
    if float(entry_px) <= 0:
        return None
    return float(exit_px / entry_px - 1.0)


def compute_forward_outcomes_for_snapshots(
    snapshot_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compute 5/10/20-session forward returns, SPY excess, and cohort percentiles
    for rows sharing snapshot_date.
    """
    if not snapshot_rows:
        return []

    calendar = _trading_calendar()
    spy_close = load_raw("SPY")["close"].astype(float)
    spy_close.index = pd.DatetimeIndex(pd.to_datetime(spy_close.index)).normalize()

    by_date: dict[str, list[dict[str, Any]]] = {}
    for row in snapshot_rows:
        by_date.setdefault(str(row["snapshot_date"]), []).append(row)

    outcomes: list[dict[str, Any]] = []
    for _date, cohort in by_date.items():
        entry_ts = pd.Timestamp(_date).normalize()
        symbol_returns: dict[str, dict[str, float | None]] = {}

        for row in cohort:
            sym = str(row["symbol"]).upper()
            if not raw_exists(sym):
                continue
            close = load_raw(sym)["close"].astype(float)
            rets: dict[str, float | None] = {}
            for horizon, key in ((5, "5d"), (10, "10d"), (20, "20d")):
                rets[key] = _forward_simple_return(close, entry_ts, horizon, calendar)
            symbol_returns[sym] = rets

        spy_rets = {
            "5d": _forward_simple_return(spy_close, entry_ts, 5, calendar),
            "10d": _forward_simple_return(spy_close, entry_ts, 10, calendar),
            "20d": _forward_simple_return(spy_close, entry_ts, 20, calendar),
        }

        for row in cohort:
            sym = str(row["symbol"]).upper()
            if sym not in symbol_returns:
                continue
            rets = symbol_returns[sym]
            outcome: dict[str, Any] = {"snapshot_id": int(row["id"])}

            for horizon_key, prefix in (("5d", "5"), ("10d", "10"), ("20d", "20")):
                ret = rets.get(horizon_key)
                spy_ret = spy_rets.get(horizon_key)
                outcome[f"ret_{prefix}d"] = ret
                outcome[f"spy_ret_{prefix}d"] = spy_ret
                if ret is not None and spy_ret is not None:
                    outcome[f"excess_ret_{prefix}d"] = ret - spy_ret
                else:
                    outcome[f"excess_ret_{prefix}d"] = None

                cohort_vals = [
                    symbol_returns[s][horizon_key]
                    for s in symbol_returns
                    if symbol_returns[s].get(horizon_key) is not None
                ]
                if ret is not None and cohort_vals:
                    arr = np.array(cohort_vals, dtype=float)
                    outcome[f"universe_pct_rank_{prefix}d"] = float(
                        (arr < ret).mean() * 100.0
                    )
                else:
                    outcome[f"universe_pct_rank_{prefix}d"] = None

            outcomes.append(outcome)

    return outcomes
