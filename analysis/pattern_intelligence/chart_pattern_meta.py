"""Pattern metadata and qualification checklist for chart overlays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analysis.pattern_intelligence.candlestick_engine import CandlestickPatternHit

PATTERN_BAR_SPAN: dict[str, int] = {
    "hammer": 1,
    "doji": 1,
    "shooting_star": 1,
    "bullish_engulfing": 2,
    "bearish_engulfing": 2,
    "morning_star": 3,
    "evening_star": 3,
    "three_white_soldiers": 3,
    "three_black_crows": 3,
}


def build_pattern_metadata(
    ohlcv: pd.DataFrame,
    pattern: CandlestickPatternHit,
    *,
    structure_bias: str,
) -> dict[str, Any]:
    span = PATTERN_BAR_SPAN.get(pattern.pattern_id, 1)
    end_index = pattern.bar_index
    start_index = max(0, end_index - span + 1)
    candle_indexes = list(range(start_index, end_index + 1))
    dates = [pd.Timestamp(ohlcv.index[i]).strftime("%Y-%m-%d") for i in candle_indexes]

    checks = _qualification_checks(ohlcv, pattern, structure_bias=structure_bias)
    quality = _quality_score(pattern.strength, checks)

    annotations = _pattern_annotations(ohlcv, pattern, candle_indexes)

    return {
        "pattern_id": pattern.pattern_id,
        "label": pattern.label,
        "direction": pattern.direction,
        "confidence": round(pattern.strength, 3),
        "quality_score": quality,
        "candle_indexes": candle_indexes,
        "start_date": dates[0],
        "end_date": dates[-1],
        "qualification_checks": checks,
        "explanation": _pattern_explanation(pattern, checks, quality),
        "annotations": annotations,
        "highlighted_candles": [
            {
                "bar_index": idx,
                "date": pd.Timestamp(ohlcv.index[idx]).strftime("%Y-%m-%d"),
                "pattern_id": pattern.pattern_id,
                "role": "pattern",
            }
            for idx in candle_indexes
        ],
    }


def _quality_score(strength: float, checks: list[dict[str, Any]]) -> int:
    passed = sum(1 for check in checks if check["passed"])
    total = max(len(checks), 1)
    checklist_ratio = passed / total
    return int(round(np.clip(strength * 0.55 + checklist_ratio * 0.45, 0.05, 1.0) * 100))


def _pattern_explanation(
    pattern: CandlestickPatternHit,
    checks: list[dict[str, Any]],
    quality: int,
) -> str:
    passed_labels = [c["label"] for c in checks if c["passed"]]
    body = "; ".join(passed_labels[:4])
    return f"{pattern.label}: {body}. Quality {quality}/100."


def _qualification_checks(
    ohlcv: pd.DataFrame,
    pattern: CandlestickPatternHit,
    *,
    structure_bias: str,
) -> list[dict[str, Any]]:
    idx = pattern.bar_index
    o = float(ohlcv["open"].iloc[idx])
    h = float(ohlcv["high"].iloc[idx])
    l = float(ohlcv["low"].iloc[idx])
    c = float(ohlcv["close"].iloc[idx])
    prev_bull = idx > 0 and float(ohlcv["close"].iloc[idx - 1]) > float(
        ohlcv["open"].iloc[idx - 1]
    )
    prev_bear = idx > 0 and float(ohlcv["close"].iloc[idx - 1]) < float(
        ohlcv["open"].iloc[idx - 1]
    )
    ret_20 = (
        float(ohlcv["close"].pct_change(20).iloc[idx])
        if idx >= 20
        else float("nan")
    )

    checks: list[dict[str, Any]] = []

    if pattern.pattern_id == "bearish_engulfing":
        checks = [
            _check("Previous candle bullish", prev_bull),
            _check("Current candle bearish", c < o),
            _check(
                "Current body fully engulfs prior body",
                idx > 0
                and c <= float(ohlcv["open"].iloc[idx - 1])
                and o >= float(ohlcv["close"].iloc[idx - 1]),
            ),
            _check("Occurred after an advance", ret_20 > 0.02 or structure_bias == "uptrend"),
        ]
    elif pattern.pattern_id == "bullish_engulfing":
        checks = [
            _check("Previous candle bearish", prev_bear),
            _check("Current candle bullish", c > o),
            _check(
                "Current body fully engulfs prior body",
                idx > 0
                and c >= float(ohlcv["open"].iloc[idx - 1])
                and o <= float(ohlcv["close"].iloc[idx - 1]),
            ),
            _check("Occurred after a decline", ret_20 < -0.02 or structure_bias == "downtrend"),
        ]
    elif pattern.pattern_id == "hammer":
        body = abs(c - o)
        rng = max(h - l, 1e-9)
        lower = min(o, c) - l
        checks = [
            _check("Small real body", body / rng <= 0.35),
            _check("Long lower shadow", lower >= body * 2),
            _check("Appears after weakness", ret_20 <= 0.03 or structure_bias != "uptrend"),
        ]
    elif pattern.pattern_id == "shooting_star":
        body = abs(c - o)
        rng = max(h - l, 1e-9)
        upper = h - max(o, c)
        checks = [
            _check("Small real body near lows", body / rng <= 0.35),
            _check("Long upper shadow", upper >= body * 2),
            _check("Appears after advance", ret_20 >= 0.02 or structure_bias == "uptrend"),
        ]
    elif pattern.pattern_id == "doji":
        body = abs(c - o)
        rng = max(h - l, 1e-9)
        checks = [
            _check("Open and close nearly equal", body / rng <= 0.1),
            _check("Indecision candle", rng > 0),
        ]
    elif pattern.pattern_id in {"morning_star", "evening_star"} and idx >= 2:
        first = float(ohlcv["close"].iloc[idx - 2]) - float(ohlcv["open"].iloc[idx - 2])
        third = float(ohlcv["close"].iloc[idx]) - float(ohlcv["open"].iloc[idx])
        if pattern.pattern_id == "morning_star":
            checks = [
                _check("First candle bearish", first < 0),
                _check("Third candle bullish", third > 0),
                _check("Reversal after decline", ret_20 <= 0.05),
            ]
        else:
            checks = [
                _check("First candle bullish", first > 0),
                _check("Third candle bearish", third < 0),
                _check("Reversal after advance", ret_20 >= -0.05),
            ]
    elif pattern.pattern_id == "three_white_soldiers" and idx >= 2:
        closes = [float(ohlcv["close"].iloc[idx - i]) for i in range(2, -1, -1)]
        checks = [
            _check("Three consecutive up closes", closes[0] < closes[1] < closes[2]),
            _check("Each session bullish", all(
                float(ohlcv["close"].iloc[idx - i]) > float(ohlcv["open"].iloc[idx - i])
                for i in range(2, -1, -1)
            )),
        ]
    elif pattern.pattern_id == "three_black_crows" and idx >= 2:
        closes = [float(ohlcv["close"].iloc[idx - i]) for i in range(2, -1, -1)]
        checks = [
            _check("Three consecutive down closes", closes[0] > closes[1] > closes[2]),
            _check("Each session bearish", all(
                float(ohlcv["close"].iloc[idx - i]) < float(ohlcv["open"].iloc[idx - i])
                for i in range(2, -1, -1)
            )),
        ]
    else:
        checks = [_check("Pattern geometry detected", pattern.strength >= 0.2)]

    return checks


def _check(label: str, passed: bool) -> dict[str, Any]:
    return {"label": label, "passed": bool(passed)}


def _pattern_annotations(
    ohlcv: pd.DataFrame,
    pattern: CandlestickPatternHit,
    candle_indexes: list[int],
) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    key_index = candle_indexes[-1]
    key_price = float(ohlcv["high"].iloc[key_index])
    if pattern.direction == "bearish":
        key_price = float(ohlcv["high"].iloc[key_index]) * 1.002
        position = "aboveBar"
    elif pattern.direction == "bullish":
        key_price = float(ohlcv["low"].iloc[key_index]) * 0.998
        position = "belowBar"
    else:
        key_price = float(ohlcv["close"].iloc[key_index])
        position = "aboveBar"

    annotations.append(
        {
            "type": "arrow",
            "bar_index": key_index,
            "date": pd.Timestamp(ohlcv.index[key_index]).strftime("%Y-%m-%d"),
            "price": round(key_price, 2),
            "label": pattern.label,
            "direction": pattern.direction,
            "position": position,
        }
    )

    for idx in candle_indexes[:-1]:
        annotations.append(
            {
                "type": "marker",
                "bar_index": idx,
                "date": pd.Timestamp(ohlcv.index[idx]).strftime("%Y-%m-%d"),
                "price": round(float(ohlcv["close"].iloc[idx]), 2),
                "label": "Pattern leg",
                "position": "inBar",
            }
        )
    return annotations
