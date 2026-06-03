import numpy as np
import pandas as pd

from app.builders.emerging_leaders_engine import (
    evaluate_emerging_leader,
    passes_emerging_leader_list,
)


def _consolidation_frame(rows: int = 130, drift: float = 0.02) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.full(rows, 100.0) + np.sin(np.linspace(0, 10 * np.pi, rows)) * 0.35
    close = close + np.linspace(0, drift, rows)
    spread = np.linspace(1.2, 0.45, rows)
    high = close + spread * 0.5
    low = close - spread * 0.5
    volume = np.full(rows, 900_000.0)
    volume[-40:-8] *= 0.55
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_quiet_consolidation_passes_purity_filter():
    frame = _consolidation_frame()
    cap = float(frame["high"].tail(60).max())
    frame.loc[frame.index[-1], ["high", "close", "open"]] = [
        cap * 0.985,
        cap * 0.965,
        cap * 0.965,
    ]
    result = evaluate_emerging_leader("TEST", frame)
    assert result is not None
    assert result.components.setup_purity_score >= 38
    assert not result.components.momentum_leader_like
    assert passes_emerging_leader_list(result)
    assert result.setup_stage in {"BASE_BUILDING", "TIGHTENING", "BREAKOUT_WATCH"}


def test_momentum_run_rejected():
    frame = _consolidation_frame()
    n = 25
    start = float(frame["close"].iloc[-n - 1])
    frame.loc[frame.index[-n:], "close"] = np.linspace(start, start * 1.16, n)
    frame.loc[frame.index[-n:], "high"] = frame["close"] + 0.4
    frame.loc[frame.index[-n:], "low"] = frame["close"] - 0.4
    result = evaluate_emerging_leader("TEST", frame)
    assert result is None or not passes_emerging_leader_list(result)


def test_extended_stage_filtered_from_list():
    frame = _consolidation_frame(drift=0.15)
    frame.loc[frame.index[-15:], "close"] = np.linspace(
        float(frame["close"].iloc[-16]),
        float(frame["close"].iloc[-16]) * 1.14,
        15,
    )
    frame.loc[frame.index[-15:], "high"] = frame["close"] + 0.5
    frame.loc[frame.index[-15:], "low"] = frame["close"] - 0.5
    result = evaluate_emerging_leader("TEST", frame)
    if result is not None:
        assert not passes_emerging_leader_list(result) or result.setup_stage != "EXTENDED"


def test_tightening_stage_requires_compression():
    frame = _consolidation_frame()
    cap = float(frame["high"].tail(60).max())
    for i in range(-25, 0):
        frame.loc[frame.index[i], "high"] = cap * 0.99
        frame.loc[frame.index[i], "close"] = cap * 0.94
        frame.loc[frame.index[i], "low"] = cap * 0.91
    frame.loc[frame.index[-5:], "high"] = cap * 0.988
    frame.loc[frame.index[-5:], "close"] = cap * 0.965
    result = evaluate_emerging_leader("TEST", frame)
    assert result is not None
    assert result.components.dormancy_days >= 8
    assert result.components.base_age >= 15
