import os

# Default test secret when JWT is used during tests.
os.environ.setdefault(
    "JWT_SECRET_KEY",
    "test-jwt-secret-key-for-pytest-only-32chars",
)


def seed_pattern_benchmarks(rows: int = 600) -> None:
    """Seed SPY and ^VIX raw OHLCV for pattern-model tests."""
    import numpy as np
    import pandas as pd

    from data.benchmarks import BENCHMARK_SYMBOL, VIX_SYMBOL
    from data.store import OHLCV_COLUMNS, save_raw

    index = pd.date_range("2020-01-01", periods=rows, freq="B", name="date")
    spy_close = np.linspace(300.0, 420.0, rows)
    vix_close = 18.0 + np.sin(np.linspace(0.0, 12.0, rows)) * 3.0

    def frame(close: np.ndarray, volume: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": close * 0.998,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": [volume] * rows,
            },
            index=index,
        ).loc[:, list(OHLCV_COLUMNS)]

    save_raw(frame(spy_close, 50_000_000), BENCHMARK_SYMBOL)
    save_raw(frame(vix_close, 1_000_000), VIX_SYMBOL)
