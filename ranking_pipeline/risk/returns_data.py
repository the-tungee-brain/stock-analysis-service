"""Load historical daily returns for risk calculations (read-only OHLCV)."""

from __future__ import annotations

import pandas as pd

from data.store import load_raw


def load_daily_returns(
    symbols: list[str],
    as_of: pd.Timestamp,
    *,
    benchmark_symbol: str = "SPY",
    lookback_days: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return aligned daily simple returns for symbols + benchmark.

    Columns = symbols; only dates ≤ ``as_of``; tail ``lookback_days`` rows.
    """
    frames: dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            raw = load_raw(symbol)
        except FileNotFoundError:
            continue
        raw = raw[raw.index <= as_of].tail(lookback_days + 1)
        if len(raw) < 10:
            continue
        frames[symbol.strip().upper()] = raw["close"].astype("float64").pct_change()

    spy_ret: pd.Series | None = None
    try:
        spy = load_raw(benchmark_symbol)
        spy = spy[spy.index <= as_of].tail(lookback_days + 1)
        spy_ret = spy["close"].astype("float64").pct_change()
        spy_ret.name = benchmark_symbol
    except FileNotFoundError:
        spy_ret = None

    if not frames:
        return pd.DataFrame(), pd.Series(dtype="float64")

    returns = pd.DataFrame(frames).dropna(how="any")
    if spy_ret is not None:
        spy_aligned = spy_ret.reindex(returns.index).dropna()
        returns = returns.loc[spy_aligned.index]
        spy_aligned = spy_aligned.reindex(returns.index)
    else:
        spy_aligned = pd.Series(0.0, index=returns.index)

    return returns, spy_aligned
