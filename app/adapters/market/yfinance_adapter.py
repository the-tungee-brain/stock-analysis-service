import yfinance as yf
import pandas as pd


class YFinanceAdapter:
    PEER_INFO_KEYS = ("recommendedSymbols", "recommended_symbols")

    def get_daily_closes_1y(self, symbol: str) -> pd.Series:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y", interval="1d")
        return hist["Close"] if "Close" in hist.columns else pd.Series(dtype=float)

    def get_ticker_info(self, symbol: str) -> dict:
        ticker = yf.Ticker(symbol)
        return ticker.info or {}

    def get_recommended_peers(self, symbol: str, *, limit: int = 10) -> list[str]:
        info = self.get_ticker_info(symbol=symbol)
        if not info:
            return []

        peers: list[str] = []
        seen: set[str] = set()
        symbol_upper = symbol.strip().upper()

        for key in self.PEER_INFO_KEYS:
            raw = info.get(key)
            if not isinstance(raw, list):
                continue
            for item in raw:
                if not isinstance(item, str):
                    continue
                peer = item.strip().upper()
                if not peer or peer == symbol_upper or peer in seen:
                    continue
                seen.add(peer)
                peers.append(peer)
                if len(peers) >= limit:
                    return peers

        return peers
