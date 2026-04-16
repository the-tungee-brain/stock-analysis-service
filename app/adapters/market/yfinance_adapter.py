import yfinance as yf
import pandas as pd


class YFinanceAdapter:
    def get_daily_closes_1y(self, symbol: str) -> pd.Series:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d")
        return hist["Close"] if "Close" in hist.columns else pd.Series(dtype=float)
