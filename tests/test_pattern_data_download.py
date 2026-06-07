"""Tests for daily OHLCV download and Parquet storage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.benchmarks import VIX_SYMBOL, ensure_benchmark_ohlcv
from data.download import download_and_store_symbol, download_symbol, main as download_main
from data.loader import load_symbol
from data.paths import raw_parquet_path
from data.store import OHLCV_COLUMNS, _normalize_ohlcv, save_raw


@pytest.mark.integration
def test_download_symbol_returns_normalized_ohlcv(tmp_path, monkeypatch):
    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yfinance-cache"))

    df = download_symbol("AAPL", years=2)

    assert not df.empty
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert df.index.name == "date"
    assert df.index.is_monotonic_increasing
    assert (df[["open", "high", "low", "close", "volume"]] > 0).all().all()


@pytest.mark.integration
def test_download_and_store_symbol_writes_parquet(tmp_path, monkeypatch):
    monkeypatch.setenv("YFINANCE_CACHE_DIR", str(tmp_path / "yfinance-cache"))
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)

    path, df = download_and_store_symbol("AAPL", years=2)

    assert path == raw_dir / "AAPL.parquet"
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(OHLCV_COLUMNS)


def test_load_symbol_reads_parquet(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    monkeypatch.setattr("data.paths.RAW_DIR", raw_dir)
    raw_dir.mkdir(parents=True)

    index = pd.date_range("2024-01-01", periods=5, freq="B", name="date")
    sample = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1_000_000] * 5,
        },
        index=index,
    )
    sample.to_parquet(raw_parquet_path("AAPL"))

    loaded = load_symbol("AAPL")
    assert len(loaded) == 5
    assert list(loaded.columns) == list(OHLCV_COLUMNS)


def test_normalize_ohlcv_drops_incomplete_current_day_row():
    index = pd.date_range("2026-06-02", periods=3, freq="B", name="date")
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0, None],
            "High": [102.0, 103.0, None],
            "Low": [99.0, 100.0, None],
            "Close": [101.0, 102.0, None],
            "Volume": [1_000_000, 1_100_000, 900_000],
        },
        index=index,
    )

    normalized = _normalize_ohlcv(raw)

    assert list(normalized.columns) == list(OHLCV_COLUMNS)
    assert len(normalized) == 2
    assert normalized.index.max() == index[1]
    assert (normalized.loc[:, list(OHLCV_COLUMNS)] > 0).all().all()


def _zero_volume_yahoo_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=3, freq="B", name="date")
    return pd.DataFrame(
        {
            "Open": [18.0, 18.5, 19.0],
            "High": [18.8, 19.0, 19.3],
            "Low": [17.7, 18.1, 18.6],
            "Close": [18.4, 18.8, 18.9],
            "Volume": [0, 0, 0],
        },
        index=index,
    )


def _sample_ohlcv(
    *,
    rows: int = 5,
    start: str = "2024-01-01",
    close: float = 100.0,
    volume: int = 1_000_000,
) -> pd.DataFrame:
    prices = [close + float(i) for i in range(rows)]
    return pd.DataFrame(
        {
            "open": [price * 0.998 for price in prices],
            "high": [price * 1.01 for price in prices],
            "low": [price * 0.99 for price in prices],
            "close": prices,
            "volume": [volume] * rows,
        },
        index=pd.date_range(start, periods=rows, freq="B", name="date"),
    )


def test_save_raw_empty_vix_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_sample_ohlcv(rows=10, close=18.0, volume=0), VIX_SYMBOL)
    before = load_symbol(VIX_SYMBOL)

    empty = pd.DataFrame(columns=list(OHLCV_COLUMNS), index=pd.DatetimeIndex([], name="date"))
    with pytest.raises(ValueError, match="Refusing to write empty raw OHLCV"):
        save_raw(empty, VIX_SYMBOL)

    after = load_symbol(VIX_SYMBOL)
    assert len(after) == len(before)
    assert after["close"].iloc[-1] == pytest.approx(before["close"].iloc[-1])


def test_save_raw_all_filtered_equity_does_not_overwrite_existing_file(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_sample_ohlcv(rows=5, close=100.0, volume=1_000_000), "AAPL")
    before = load_symbol("AAPL")

    with pytest.raises(ValueError, match="Refusing to write empty raw OHLCV"):
        save_raw(_sample_ohlcv(rows=3, close=101.0, volume=0), "AAPL")

    after = load_symbol("AAPL")
    assert len(after) == len(before)
    assert after["close"].iloc[-1] == pytest.approx(before["close"].iloc[-1])


def test_save_raw_rejects_truncated_vix_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_sample_ohlcv(rows=100, start="2024-01-01", close=18.0, volume=0), VIX_SYMBOL)
    before = load_symbol(VIX_SYMBOL)

    with pytest.raises(ValueError, match="truncated data"):
        save_raw(_sample_ohlcv(rows=5, start="2024-05-13", close=20.0, volume=0), VIX_SYMBOL)

    after = load_symbol(VIX_SYMBOL)
    assert len(after) == len(before)
    assert after.index.max() == before.index.max()


def test_save_raw_merges_newer_valid_vix_data(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    save_raw(_sample_ohlcv(rows=10, start="2024-01-01", close=18.0, volume=0), VIX_SYMBOL)

    save_raw(_sample_ohlcv(rows=2, start="2024-01-15", close=28.0, volume=0), VIX_SYMBOL)

    loaded = load_symbol(VIX_SYMBOL)
    assert len(loaded) == 12
    assert loaded.index.max() == pd.Timestamp("2024-01-16")
    assert loaded["close"].iloc[-1] == pytest.approx(29.0)


def test_save_raw_writes_readable_parquet_permissions(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)

    path = save_raw(_sample_ohlcv(rows=3, close=100.0, volume=1_000_000), "AAPL")

    assert path.stat().st_mode & 0o777 == 0o644


def test_download_symbol_preserves_zero_volume_for_vix(monkeypatch):
    monkeypatch.setattr("data.download.configure_yfinance", lambda: None)
    monkeypatch.setattr("data.download.yf.download", lambda *args, **kwargs: _zero_volume_yahoo_frame())

    df = download_symbol(VIX_SYMBOL, years=1)

    assert len(df) == 3
    assert list(df.columns) == list(OHLCV_COLUMNS)
    assert (df["volume"] == 0).all()


def test_download_symbol_preserves_zero_volume_for_vix_alias(monkeypatch):
    monkeypatch.setattr("data.download.configure_yfinance", lambda: None)
    monkeypatch.setattr("data.download.yf.download", lambda *args, **kwargs: _zero_volume_yahoo_frame())

    df = download_symbol("VIX", years=1)

    assert len(df) == 3
    assert (df["volume"] == 0).all()


def test_download_symbol_filters_zero_volume_for_equities(monkeypatch):
    monkeypatch.setattr("data.download.configure_yfinance", lambda: None)
    monkeypatch.setattr("data.download.yf.download", lambda *args, **kwargs: _zero_volume_yahoo_frame())
    monkeypatch.setattr("data.download.time.sleep", lambda _: None)

    with pytest.raises(ValueError, match="No data returned for AAPL"):
        download_symbol("AAPL", years=1)


def test_download_symbol_without_retry_attempts_once(monkeypatch):
    calls = 0

    def fail_fetch(*args, **kwargs):  # noqa: ANN002, ANN003
        nonlocal calls
        calls += 1
        raise RuntimeError("rate limited")

    monkeypatch.setattr("data.download._fetch_yahoo_ohlcv", fail_fetch)
    monkeypatch.setattr("data.download.time.sleep", lambda _: pytest.fail("unexpected retry delay"))

    with pytest.raises(ValueError, match="No data returned for AAPL"):
        download_symbol("AAPL", years=1, retry=False)

    assert calls == 1


def test_download_cli_no_retry_passes_single_attempt_mode(monkeypatch, capsys):
    seen: dict[str, object] = {}

    def fake_download_and_store_all(symbols, *, years, retry):  # noqa: ANN001
        seen["symbols"] = symbols
        seen["years"] = years
        seen["retry"] = retry
        return {"SPY": _sample_ohlcv(rows=1)}

    monkeypatch.setattr("data.download.download_and_store_all", fake_download_and_store_all)

    assert download_main(["--symbols", "SPY", "--years", "1", "--no-retry"]) == 0

    assert seen == {"symbols": ["SPY"], "years": 1, "retry": False}
    assert "Saved SPY: 1 rows" in capsys.readouterr().out


def test_missing_vix_benchmark_refresh_creates_non_empty_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr("data.paths.RAW_DIR", tmp_path)
    monkeypatch.setattr("data.download.configure_yfinance", lambda: None)

    def fake_download(symbol: str, *args, **kwargs):  # noqa: ANN001
        if symbol.strip().upper() == "SPY":
            return pd.DataFrame(
                {
                    "Open": [400.0],
                    "High": [401.0],
                    "Low": [399.0],
                    "Close": [400.5],
                    "Volume": [50_000_000],
                },
                index=pd.date_range("2024-01-01", periods=1, freq="B", name="date"),
            )
        return _zero_volume_yahoo_frame()

    monkeypatch.setattr("data.download.yf.download", fake_download)

    ensure_benchmark_ohlcv(years=1)

    loaded = load_symbol(VIX_SYMBOL)
    assert len(loaded) == 3
    assert (loaded["volume"] == 0).all()
    assert raw_parquet_path(VIX_SYMBOL).exists()
