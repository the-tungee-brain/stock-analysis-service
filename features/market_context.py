"""Cross-asset market context features (SPY, VIX, relative strength)."""

from __future__ import annotations

import pandas as pd

MOMENTUM_HORIZONS: tuple[int, ...] = (21, 63, 126, 252)


def compute_spy_market_features(spy_close: pd.Series) -> pd.DataFrame:
    """SPY returns and momentum features indexed by date."""
    close = spy_close.astype("float64")
    out = pd.DataFrame(index=close.index)
    out["spy_ret_1d"] = close.pct_change(1)
    out["spy_ret_5d"] = close.pct_change(5)
    out["spy_ret_20d"] = close.pct_change(20)
    for horizon in MOMENTUM_HORIZONS:
        out[f"spy_mom_{horizon}d"] = close.pct_change(horizon)
    return out


def compute_vix_features(vix_close: pd.Series) -> pd.DataFrame:
    """VIX level and short-term change."""
    close = vix_close.astype("float64")
    out = pd.DataFrame(index=close.index)
    out["vix_level"] = close
    out["vix_chg_5d"] = close.pct_change(5)
    return out


def compute_relative_strength_vs_spy(
    stock_close: pd.Series,
    spy_close: pd.Series,
    *,
    horizons: tuple[int, ...] = MOMENTUM_HORIZONS,
) -> pd.DataFrame:
    """Stock minus SPY cumulative return over each momentum horizon."""
    stock = stock_close.astype("float64")
    spy = spy_close.reindex(stock.index).astype("float64")
    out = pd.DataFrame(index=stock.index)
    for horizon in horizons:
        stock_ret = stock.pct_change(horizon)
        spy_ret = spy.pct_change(horizon)
        out[f"rs_vs_spy_{horizon}d"] = stock_ret - spy_ret
    return out


def attach_market_context(
    features: pd.DataFrame,
    *,
    stock_close: pd.Series,
    spy_close: pd.Series,
    vix_close: pd.Series,
) -> pd.DataFrame:
    """Join SPY, VIX, and relative-strength columns onto a symbol feature matrix."""
    spy_aligned = spy_close.reindex(features.index).astype("float64")
    vix_aligned = vix_close.reindex(features.index).astype("float64")
    stock_aligned = stock_close.reindex(features.index).astype("float64")

    context = pd.concat(
        [
            compute_spy_market_features(spy_aligned),
            compute_vix_features(vix_aligned),
            compute_relative_strength_vs_spy(stock_aligned, spy_aligned),
        ],
        axis=1,
    )
    return features.join(context, how="left")
