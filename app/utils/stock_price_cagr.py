from __future__ import annotations


def fetch_price_cagr_pct(symbol: str, *, lookback_years: int = 5) -> float | None:
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        ticker = yf.Ticker(symbol.strip().upper())
        hist = ticker.history(period=f"{lookback_years}y", interval="1d")
        if hist.empty or len(hist) < 2:
            return None

        start_value = float(hist["Close"].iloc[0])
        end_value = float(hist["Close"].iloc[-1])
        if start_value <= 0 or end_value <= 0:
            return None

        elapsed_years = (hist.index[-1] - hist.index[0]).days / 365.25
        if elapsed_years < 1:
            return None

        cagr = (end_value / start_value) ** (1 / elapsed_years) - 1
        return round(cagr * 100.0, 2)
    except Exception:
        return None
