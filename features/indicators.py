"""Technical indicators via pandas-ta."""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, moving averages, MACD, Bollinger Bands, and ATR."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = ta.rsi(close, length=14)

    for length in (20, 50, 200):
        out[f"sma_{length}"] = ta.sma(close, length=length)

    for length in (12, 26):
        out[f"ema_{length}"] = ta.ema(close, length=length)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        out["macd"] = macd.iloc[:, 0]
        out["macd_hist"] = macd.iloc[:, 1]
        out["macd_signal"] = macd.iloc[:, 2]

    bbands = ta.bbands(close, length=20, std=2)
    if bbands is not None and not bbands.empty:
        out["bb_lower"] = bbands.iloc[:, 0]
        out["bb_mid"] = bbands.iloc[:, 1]
        out["bb_upper"] = bbands.iloc[:, 2]
        out["bb_pct"] = bbands.iloc[:, 4]

    out["atr_14"] = ta.atr(high, low, close, length=14)
    return out
