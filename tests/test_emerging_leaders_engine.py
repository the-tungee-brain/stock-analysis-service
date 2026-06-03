import numpy as np
import pandas as pd

from app.builders.emerging_leaders_engine import evaluate_emerging_leader


def _synthetic_base_frame(rows: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = np.linspace(100, 108, rows)
    noise = np.sin(np.linspace(0, 8 * np.pi, rows)) * 0.4
    close = close + noise
    high = close + 0.6
    low = close - 0.6
    volume = np.full(rows, 1_000_000.0)
    volume[-30:-10] *= 0.65
    volume[-5:] *= 1.4
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_evaluate_tightening_or_watch_stage():
    frame = _synthetic_base_frame()
    frame.loc[frame.index[-1], "high"] = frame["close"].iloc[-1] + 0.2
    result = evaluate_emerging_leader("TEST", frame)
    assert result is not None
    assert 0 <= result.setup_quality_score <= 100
    assert result.setup_stage in {
        "BASE_BUILDING",
        "TIGHTENING",
        "BREAKOUT_WATCH",
        "BREAKOUT_TRIGGERED",
        "EXTENDED",
    }
    assert result.positive_factors
    assert result.missing_factors
    assert result.next_confirmation


def test_extended_stage_on_large_recent_move():
    frame = _synthetic_base_frame()
    frame.loc[frame.index[-20:], "close"] = np.linspace(
        frame["close"].iloc[-21],
        frame["close"].iloc[-21] * 1.18,
        20,
    )
    frame.loc[frame.index[-20:], "high"] = frame["close"] + 0.5
    frame.loc[frame.index[-20:], "low"] = frame["close"] - 0.5
    result = evaluate_emerging_leader("TEST", frame)
    assert result is not None
    assert result.setup_stage in {"EXTENDED", "BREAKOUT_TRIGGERED"}
