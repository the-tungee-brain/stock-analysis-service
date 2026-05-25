from __future__ import annotations

from dataclasses import dataclass

from app.adapters.sec.sec_edgar_adapter import SecEdgarAdapter
from app.data.etf_core_symbols import ETF_CORE_SYMBOLS


@dataclass(frozen=True)
class SymbolSearchEntry:
    symbol: str
    name: str


class SymbolSearchBuilder:
    def __init__(self, sec_edgar_adapter: SecEdgarAdapter):
        self.sec_edgar_adapter = sec_edgar_adapter
        self._entries: list[SymbolSearchEntry] | None = None

    def _load_entries(self) -> list[SymbolSearchEntry]:
        if self._entries is not None:
            return self._entries

        by_symbol: dict[str, SymbolSearchEntry] = {}

        raw = self.sec_edgar_adapter.get_company_tickers()
        for entry in raw.values():
            symbol = str(entry.get("ticker", "")).upper().strip()
            title = str(entry.get("title", "")).strip()
            if not symbol or not title:
                continue
            by_symbol[symbol] = SymbolSearchEntry(symbol=symbol, name=title)

        for symbol, name in ETF_CORE_SYMBOLS.items():
            key = symbol.upper()
            by_symbol.setdefault(key, SymbolSearchEntry(symbol=key, name=name))

        self._entries = sorted(by_symbol.values(), key=lambda item: item.symbol)
        return self._entries

    @staticmethod
    def _score(entry: SymbolSearchEntry, query: str) -> int:
        symbol = entry.symbol
        name = entry.name.upper()
        query_upper = query.upper()
        query_words = [word for word in query_upper.split() if word]

        if symbol == query_upper:
            return 100
        if symbol.startswith(query_upper):
            return 90
        if query_upper in symbol:
            return 80
        if name.startswith(query_upper):
            return 70
        if query_words and all(word in name for word in query_words):
            return 60
        if query_upper in name:
            return 50
        return 0

    def search(self, keyword: str, *, limit: int = 10) -> list[SymbolSearchEntry]:
        query = keyword.strip()
        if not query:
            return []

        resolved_limit = max(1, min(limit, 100))
        scored: list[tuple[int, SymbolSearchEntry]] = []

        for entry in self._load_entries():
            score = self._score(entry, query)
            if score <= 0:
                continue
            scored.append((score, entry))

        scored.sort(key=lambda item: (-item[0], item[1].symbol))
        return [entry for _, entry in scored[:resolved_limit]]
