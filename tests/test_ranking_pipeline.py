"""Tests for ranking pipeline: features, composite, storage, incremental OHLCV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.store import append_raw, merge_ohlcv, raw_exists
from ranking_pipeline.config import RankingPipelineConfig
from ranking_pipeline.features.ranking_features import compute_ranking_features
from ranking_pipeline.providers.symbol_metadata import SymbolMetadata
from ranking_pipeline.scoring.composite import score_universe_slice
from ranking_pipeline.storage.oracle_screening import ScreenResultCounts, ScreenRun
from ranking_pipeline.storage.sqlite import RankingStore
from ranking_pipeline.universe.filters import compute_adv_dollars, screen_symbol_ohlcv


def _synthetic_ohlcv(rows: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    close = 100 + np.cumsum(rng.normal(0, 1, size=rows))
    high = close + rng.uniform(0.5, 2, size=rows)
    low = close - rng.uniform(0.5, 2, size=rows)
    open_ = close + rng.normal(0, 0.5, size=rows)
    volume = rng.integers(1_000_000, 5_000_000, size=rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


def _spy_close(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(100.0, index=index)


def test_compute_ranking_features_columns():
    ohlcv = _synthetic_ohlcv()
    feats = compute_ranking_features(ohlcv, _spy_close(ohlcv.index), include_labels=True)
    assert not feats.empty
    for col in (
        "close_vs_sma20",
        "excess_ret_5d_vs_spy",
        "rel_volume",
        "dist_20d_high",
        "atr_14",
        "pattern_signal_score",
    ):
        assert col in feats.columns


def test_composite_weights_contributions_sum():
    panel = pd.DataFrame(
        {
            "close_vs_sma20": [0.1, -0.2, 0.05],
            "close_vs_sma50": [0.2, 0.0, 0.1],
            "close_vs_sma200": [0.0, 0.1, -0.1],
            "sma20_slope_5d": [0.01, 0.02, -0.01],
            "sma50_slope_5d": [0.02, -0.01, 0.0],
            "excess_ret_5d_vs_spy": [0.03, 0.04, -0.02],
            "excess_ret_20d_vs_spy": [0.05, -0.03, 0.01],
            "excess_ret_60d_vs_spy": [0.08, 0.02, 0.0],
            "rel_volume": [1.2, 0.8, 1.5],
            "vol_ratio_20d": [1.1, 0.9, 1.4],
            "dist_20d_high": [-0.02, -0.05, -0.01],
            "dist_52w_high": [-0.1, -0.2, -0.05],
            "new_high_20d": [0.0, 1.0, 0.0],
            "new_high_52w": [0.0, 0.0, 1.0],
            "atr_14": [2.0, 2.5, 1.8],
            "atr_percentile_252d": [0.5, 0.6, 0.4],
            "pat_engulfing": [0.0, 100.0, 0.0],
            "pat_hammer": [0.0, 0.0, 100.0],
            "pat_morningstar": [0.0, 0.0, 0.0],
            "pattern_signal_score": [0.0, 1.0, 1.0],
        },
        index=["AAA", "BBB", "CCC"],
    )
    cfg = RankingPipelineConfig()
    results = score_universe_slice(panel, cfg)
    weights = cfg.normalized_weights()
    for res in results:
        assert abs(sum(res.contributions.values()) - res.composite_score) < 1e-6
        for group in weights:
            if group in res.contributions:
                assert group in res.contributions


def test_liquidity_screen():
    ohlcv = _synthetic_ohlcv(30)
    ohlcv["close"] = 10.0
    ohlcv["volume"] = 5_000_000
    adv = compute_adv_dollars(ohlcv)
    assert adv == 10.0 * 5_000_000
    metrics = screen_symbol_ohlcv(
        "TEST",
        ohlcv,
        market_cap=2e9,
        filters=RankingPipelineConfig().liquidity,
    )
    assert metrics.passed is True


def test_sqlite_ranking_roundtrip(tmp_path: Path):
    db = tmp_path / "rank.db"
    store = RankingStore(db)
    store.save_universe_snapshot(
        "2026-01-01",
        [
            {
                "symbol": "AAPL",
                "last_close": 200.0,
                "market_cap": 3e12,
                "avg_dollar_volume_20d": 1e9,
                "passed_filters": True,
            }
        ],
    )
    assert store.load_universe_symbols("2026-01-01") == ["AAPL"]
    members = store.load_passed_universe_members("2026-01-01")
    assert len(members) == 1
    assert members[0].symbol == "AAPL"
    assert members[0].avg_dollar_volume_20d == 1e9
    store.save_ranking_run(
        "run-1",
        "2026-01-02",
        "composite",
        "2026-01-01",
        [
            {
                "symbol": "AAPL",
                "rank": 1,
                "composite_score": 1.5,
                "ml_probability": 0.7,
                "expected_excess_return": 0.02,
                "final_score": 1.2,
                "contributions": {"trend": 0.5, "relative_strength": 0.7},
            }
        ],
    )
    rows = store.get_ranking_results("run-1", limit=5)
    assert rows[0]["symbol"] == "AAPL"
    assert rows[0]["contributions"]["trend"] == 0.5
    with_rank = store.load_passed_universe_members("2026-01-01")
    assert with_rank[0].ranking_score == 1.2
    ordered = store.load_ranking_results_ordered("run-1")
    assert ordered[0].symbol == "AAPL"
    assert ordered[0].rank == 1
    latest = store.get_latest_ranking_run()
    assert latest is not None
    assert latest.run_id == "run-1"


def test_sqlite_universe_snapshot_incremental_finalize(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from data import paths

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    db = tmp_path / "rank.db"
    store = RankingStore(db)

    store.start_universe_snapshot("snap-1")
    store.append_universe_members(
        "snap-1",
        [
            {
                "symbol": "AAA",
                "last_close": 10.0,
                "market_cap": 2e9,
                "avg_dollar_volume_20d": 50e6,
                "passed_filters": True,
            }
        ],
    )
    store.append_universe_members(
        "snap-1",
        [{"symbol": "BBB", "passed_filters": False}],
    )

    assert store.finalize_universe_snapshot("snap-1") == 1
    assert store.load_universe_symbols("snap-1") == ["AAA"]
    assert store.active_snapshot_id() == "snap-1"


def test_refresh_universe_persists_batches_without_retaining_all_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from data import paths
    from ranking_pipeline.pipeline import weekly_universe

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    store = RankingStore(tmp_path / "rank.db")
    monkeypatch.setattr(weekly_universe, "open_store", lambda _cfg: store)
    monkeypatch.setattr(
        weekly_universe,
        "fetch_all_us_equity_symbols",
        lambda: ["AAA", "BBB", "CCC", "DDD", "EEE"],
    )
    monkeypatch.setattr(weekly_universe, "_rss_mb", lambda: 123.0)

    seen_market_caps: dict[str, float | None] = {}

    def fake_screen_one(
        symbol: str,
        _filters,
        _lookback_days: int,
        market_cap: float | None = None,
    ):
        seen_market_caps[symbol] = market_cap
        return {
            "symbol": symbol,
            "last_close": 10.0,
            "market_cap": market_cap,
            "avg_dollar_volume_20d": 50e6,
            "passed_filters": symbol in {"AAA", "CCC", "EEE"},
        }

    monkeypatch.setattr(weekly_universe, "_screen_one", fake_screen_one)
    screen_store = _FakeScreenStore()
    metadata_resolver = _FakeMetadataResolver(
        {
            "AAA": 2e9,
            "BBB": 3e9,
            "CCC": 4e9,
            "DDD": 5e9,
            "EEE": 6e9,
        }
    )

    snapshot_id = weekly_universe.refresh_universe(
        batch_size=2,
        max_workers=1,
        memory_log_interval=2,
        commit_interval=2,
        resume=False,
        screen_store=screen_store,
        metadata_resolver=metadata_resolver,
    )

    assert screen_store.upserted_symbols == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert screen_store.commit_progress_counts == [2, 4, 5]
    assert screen_store.finalized == ("run-1", 5, 3)
    assert metadata_resolver.calls == [["AAA", "BBB"], ["CCC", "DDD"], ["EEE"]]
    assert seen_market_caps == {"AAA": 2e9, "BBB": 3e9, "CCC": 4e9, "DDD": 5e9, "EEE": 6e9}
    assert screen_store.full_completed_load_called is False
    assert screen_store.full_results_load_called is False
    assert store.load_universe_symbols(snapshot_id) == ["AAA", "CCC", "EEE"]


def test_refresh_universe_resumes_completed_oracle_symbols(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from data import paths
    from ranking_pipeline.pipeline import weekly_universe

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    store = RankingStore(tmp_path / "rank.db")
    monkeypatch.setattr(weekly_universe, "open_store", lambda _cfg: store)
    monkeypatch.setattr(
        weekly_universe,
        "fetch_all_us_equity_symbols",
        lambda: ["AAA", "BBB", "CCC"],
    )
    monkeypatch.setattr(weekly_universe, "_rss_mb", lambda: 123.0)
    screened: list[str] = []

    def fake_screen_one(symbol: str, _filters, _lookback_days: int, market_cap: float | None = None):
        screened.append(symbol)
        return {
            "symbol": symbol,
            "last_close": 10.0,
            "market_cap": market_cap,
            "avg_dollar_volume_20d": 50e6,
            "passed_filters": True,
        }

    monkeypatch.setattr(weekly_universe, "_screen_one", fake_screen_one)
    screen_store = _FakeScreenStore(existing=["AAA"])
    metadata_resolver = _FakeMetadataResolver({"BBB": 3e9, "CCC": 4e9})

    snapshot_id = weekly_universe.refresh_universe(
        batch_size=2,
        max_workers=1,
        commit_interval=2,
        screen_store=screen_store,
        metadata_resolver=metadata_resolver,
    )

    assert screened == ["BBB", "CCC"]
    assert screen_store.upserted_symbols == ["BBB", "CCC"]
    assert screen_store.completed_queries == [["AAA", "BBB"], ["CCC"]]
    assert metadata_resolver.calls == [["BBB"], ["CCC"]]
    assert screen_store.full_completed_load_called is False
    assert screen_store.full_results_load_called is False
    assert store.load_universe_symbols(snapshot_id) == ["AAA", "BBB", "CCC"]


def test_refresh_universe_refuses_zero_pass_oracle_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from data import paths
    from ranking_pipeline.pipeline import weekly_universe

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    store = RankingStore(tmp_path / "rank.db")
    monkeypatch.setattr(weekly_universe, "open_store", lambda _cfg: store)
    monkeypatch.setattr(weekly_universe, "fetch_all_us_equity_symbols", lambda: ["AAA"])
    monkeypatch.setattr(weekly_universe, "_rss_mb", lambda: 123.0)
    monkeypatch.setattr(
        weekly_universe,
        "_screen_one",
        lambda *_args, **_kwargs: {
            "symbol": "AAA",
            "last_close": 10.0,
            "market_cap": None,
            "avg_dollar_volume_20d": 50e6,
            "passed_filters": False,
        },
    )
    screen_store = _FakeScreenStore()
    metadata_resolver = _FakeMetadataResolver({})

    with pytest.raises(RuntimeError, match="zero passed symbols"):
        weekly_universe.refresh_universe(
            batch_size=1,
            max_workers=1,
            resume=False,
            screen_store=screen_store,
            metadata_resolver=metadata_resolver,
        )

    assert screen_store.failed == ("run-1", 1, 0)
    assert screen_store.finalized is None
    assert store.active_snapshot_id() is None


def test_daily_restores_active_universe_from_completed_oracle_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from data import paths
    from ranking_pipeline.pipeline import daily

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    store = RankingStore(tmp_path / "rank.db")
    screen_store = _FakeScreenStore(existing=["AAA", "BBB", "CCC"])
    monkeypatch.setattr(daily, "open_oracle_screening_store", lambda: screen_store)

    symbols = daily._restore_active_universe_from_oracle(store, page_size=2)

    assert symbols == ["AAA", "BBB", "CCC"]
    assert store.active_snapshot_id() == "snap-1"
    assert store.load_universe_symbols("snap-1") == ["AAA", "BBB", "CCC"]
    assert screen_store.iter_result_page_sizes == [2]


def test_daily_ignores_completed_oracle_run_without_passed_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from data import paths
    from ranking_pipeline.pipeline import daily

    monkeypatch.setattr(paths, "RANKING_DIR", tmp_path / "ranking")
    store = RankingStore(tmp_path / "rank.db")
    screen_store = _FakeScreenStore(existing=["AAA"])
    screen_store.results["AAA"]["passed_filters"] = False
    monkeypatch.setattr(daily, "open_oracle_screening_store", lambda: screen_store)

    assert daily._restore_active_universe_from_oracle(store, page_size=2) == []
    assert store.active_snapshot_id() is None
    assert screen_store.iter_result_page_sizes == []


def test_oracle_screening_merges_expose_all_bind_placeholders():
    from ranking_pipeline.storage.oracle_screening import OracleScreeningStore

    cursor = _RecordingCursor()
    store = OracleScreeningStore(_RecordingPool(cursor))
    store.upsert_result(
        "run-1",
        {
            "symbol": "AAA",
            "last_close": 10.0,
            "market_cap": 2e9,
            "avg_dollar_volume_20d": 50e6,
            "passed_filters": True,
            "reasons": {"min_price": True},
        },
    )
    store.upsert_error("run-1", "BBB", "rate limited")
    store.update_progress(
        "run-1",
        processed_count=2,
        passed_count=1,
        rss_mb=123.0,
        commit=True,
    )

    for sql, params in cursor.executions:
        missing = [key for key in params if f":{key}" not in sql]
        assert missing == []


def test_oracle_screening_iter_results_uses_null_initial_cursor():
    from ranking_pipeline.storage.oracle_screening import OracleScreeningStore

    cursor = _PagedResultsCursor()
    store = OracleScreeningStore(_RecordingPool(cursor))

    pages = list(store.iter_results("run-1", page_size=2, passed_only=True))

    assert [row["symbol"] for page in pages for row in page] == ["AAA", "BBB"]
    assert cursor.last_symbol_binds == [None, "BBB"]
    assert "(:last_symbol IS NULL OR symbol > :last_symbol)" in cursor.executions[0][0]


class _FakeScreenStore:
    def __init__(self, existing: list[str] | None = None) -> None:
        self.run_id = "run-1"
        self.snapshot_id = "snap-1"
        self.results = {
            symbol: {
                "symbol": symbol,
                "last_close": 10.0,
                "market_cap": 2e9,
                "avg_dollar_volume_20d": 50e6,
                "passed_filters": True,
            }
            for symbol in (existing or [])
        }
        self.upserted_symbols: list[str] = []
        self.commit_progress_counts: list[int] = []
        self.finalized: tuple[str, int, int] | None = None
        self.failed: tuple[str, int, int] | None = None
        self.completed_queries: list[list[str]] = []
        self.iter_result_page_sizes: list[int] = []
        self.full_completed_load_called = False
        self.full_results_load_called = False

    def start_or_resume_run(self, **_kwargs) -> ScreenRun:
        return ScreenRun(
            run_id=self.run_id,
            snapshot_id=self.snapshot_id,
            processed_count=len(self.results),
            passed_count=sum(1 for result in self.results.values() if result["passed_filters"]),
        )

    def latest_completed_run(self) -> ScreenRun | None:
        passed = sum(1 for result in self.results.values() if result["passed_filters"])
        if passed <= 0:
            return None
        return ScreenRun(
            run_id=self.run_id,
            snapshot_id=self.snapshot_id,
            processed_count=len(self.results),
            passed_count=passed,
        )

    def completed_symbols(self, _run_id: str) -> set[str]:
        self.full_completed_load_called = True
        raise AssertionError("full completed result load should not be used")

    def completed_symbols_for(self, _run_id: str, symbols: list[str]) -> set[str]:
        self.completed_queries.append(list(symbols))
        return {symbol for symbol in symbols if symbol in self.results}

    def upsert_result(self, _run_id: str, result: dict) -> None:
        self.upserted_symbols.append(result["symbol"])
        self.results[result["symbol"]] = result

    def upsert_error(self, _run_id: str, _symbol: str, _error: str) -> None:
        return None

    def update_progress(
        self,
        _run_id: str,
        *,
        processed_count: int,
        passed_count: int,
        rss_mb: float,
        commit: bool,
    ) -> None:
        if commit:
            self.commit_progress_counts.append(processed_count)

    def mark_interrupted(self, _run_id: str, *, processed_count: int, passed_count: int) -> None:
        return None

    def mark_failed(self, run_id: str, *, processed_count: int, passed_count: int) -> None:
        self.failed = (run_id, processed_count, passed_count)

    def finalize_run(self, run_id: str, *, processed_count: int, passed_count: int) -> None:
        self.finalized = (run_id, processed_count, passed_count)

    def result_counts(self, _run_id: str) -> ScreenResultCounts:
        return ScreenResultCounts(
            total_count=len(self.results),
            passed_count=sum(1 for result in self.results.values() if result["passed_filters"]),
        )

    def load_results(self, _run_id: str) -> list[dict]:
        self.full_results_load_called = True
        raise AssertionError("full result load should not be used")

    def iter_results(self, _run_id: str, *, page_size: int, passed_only: bool = False):
        self.iter_result_page_sizes.append(page_size)
        values = sorted(self.results.values(), key=lambda row: row["symbol"])
        if passed_only:
            values = [row for row in values if row["passed_filters"]]
        for idx in range(0, len(values), page_size):
            yield values[idx : idx + page_size]

    def close(self) -> None:
        return None


class _RecordingCursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, dict]] = []
        self.input_sizes: set[str] = set()

    def setinputsizes(self, **kwargs) -> None:
        self.input_sizes = set(kwargs)

    def execute(self, sql: str, params: dict | None = None):
        missing_input_sizes = [key for key in self.input_sizes if f":{key}" not in sql]
        assert missing_input_sizes == []
        self.input_sizes = set()
        if params is not None:
            self.executions.append((sql, params))
        return self

    def fetchall(self):
        return []

    def fetchone(self):
        return None


class _PagedResultsCursor(_RecordingCursor):
    def __init__(self) -> None:
        super().__init__()
        self.last_symbol_binds: list[str | None] = []
        self._rows: list[tuple] = []

    def execute(self, sql: str, params: dict | None = None):
        super().execute(sql, params)
        last_symbol = params["last_symbol"] if params else None
        self.last_symbol_binds.append(last_symbol)
        self._rows = (
            [
                ("AAA", 10.0, 2e9, 50e6, 1),
                ("BBB", 11.0, 3e9, 60e6, 1),
            ]
            if last_symbol is None
            else []
        )
        return self

    def fetchall(self):
        return self._rows


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor) -> None:
        self._cursor = cursor
        self.commits = 0

    def cursor(self) -> _RecordingCursor:
        return self._cursor

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()


class _RecordingPool:
    def __init__(self, cursor: _RecordingCursor) -> None:
        self._connection = _RecordingConnection(cursor)

    def acquire(self) -> _RecordingConnection:
        return self._connection


class _FakeMetadataResolver:
    def __init__(self, market_caps: dict[str, float]) -> None:
        self.market_caps = market_caps
        self.calls: list[list[str]] = []

    def resolve_many(self, symbols: list[str], *, required_fields):
        self.calls.append(list(symbols))
        return {
            symbol: SymbolMetadata(symbol=symbol, market_cap=self.market_caps.get(symbol))
            for symbol in symbols
        }


def test_merge_ohlcv_dedupes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dates = pd.bdate_range("2024-01-01", periods=3)
    first = pd.DataFrame(
        {
            "open": [1, 2, 3],
            "high": [2, 3, 4],
            "low": [0.5, 1.5, 2.5],
            "close": [1.5, 2.5, 3.5],
            "volume": [100, 200, 300],
        },
        index=dates,
    )
    second = first.copy()
    second.iloc[-1, second.columns.get_loc("close")] = 99.0
    merged = merge_ohlcv(first, second)
    assert merged.iloc[-1]["close"] == 99.0
    assert len(merged) == 3


def test_incremental_raw_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from data import paths

    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    tmp_path.joinpath("raw").mkdir()
    dates = pd.bdate_range("2024-01-01", periods=2)
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        },
        index=dates,
    )
    append_raw(df, "ZZZ")
    assert raw_exists("ZZZ")
    extra = df.iloc[[-1]].copy()
    extra.iloc[0, extra.columns.get_loc("close")] = 5.0
    append_raw(extra, "ZZZ")
    from data.store import load_raw

    loaded = load_raw("ZZZ")
    assert loaded.iloc[-1]["close"] == 5.0
