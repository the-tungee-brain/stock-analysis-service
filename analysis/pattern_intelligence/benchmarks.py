"""Model C benchmark symbols where excess-return vs SPY is undefined."""

from __future__ import annotations

from data.benchmarks import BENCHMARK_SYMBOL
from data.symbols import BENCHMARK_ONLY_SYMBOLS

BENCHMARK_NOTICE = (
    "This symbol is the Model C benchmark. Excess return vs SPY is always zero here, "
    "so ranking probabilities are undefined — use pattern, trend, and regime context only."
)


def is_model_benchmark_symbol(symbol: str | None) -> bool:
    if not symbol:
        return False
    upper = symbol.strip().upper()
    return upper == BENCHMARK_SYMBOL or upper in BENCHMARK_ONLY_SYMBOLS
