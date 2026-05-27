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


def fetch_dividend_yield_pct(symbol: str) -> float | None:
    try:
        import yfinance as yf
    except ImportError:
        return None

    try:
        info = yf.Ticker(symbol.strip().upper()).info
        raw = info.get("dividendYield")
        if raw is None or not isinstance(raw, (int, float)):
            return None
        value = float(raw)
        pct = value * 100.0 if abs(value) < 1 else value
        return round(pct, 2)
    except Exception:
        return None
