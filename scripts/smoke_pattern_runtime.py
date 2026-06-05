#!/usr/bin/env python3
"""Smoke-check deployed pattern model artifacts and paid pattern endpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import requests

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.services.pattern_analysis_service import PatternAnalysisService
from app.services.pattern_forecast_service import pattern_forecast_from_prediction
from app.services.pattern_intelligence_service import pattern_intelligence_from_dict
from data.benchmarks import (
    BENCHMARK_SYMBOLS,
    benchmark_raw_paths,
    ensure_benchmark_ohlcv,
    load_benchmark_ohlcv,
)
from data.paths import raw_parquet_path
from models.artifact_store import resolve_artifact_dir
from models.prediction_service import ensure_raw_ohlcv, load_deployed_model


def _require_non_empty_ohlcv(
    frame,  # noqa: ANN001
    *,
    label: str,
    paths: Sequence[str],
) -> None:
    if frame is not None and not frame.empty:
        return
    location = ", ".join(paths)
    raise RuntimeError(f"Required OHLCV is missing or empty for {label}. Expected: {location}")


def _check_runtime_ohlcv(symbol: str) -> None:
    symbol_upper = symbol.strip().upper()
    try:
        raw = ensure_raw_ohlcv(symbol_upper)
    except Exception as exc:
        raise RuntimeError(
            f"Required OHLCV is missing or empty for {symbol_upper}. "
            f"Expected: {raw_parquet_path(symbol_upper)}"
        ) from exc
    _require_non_empty_ohlcv(
        raw,
        label=symbol_upper,
        paths=[str(raw_parquet_path(symbol_upper))],
    )

    prepare_error: Exception | None = None
    try:
        ensure_benchmark_ohlcv()
    except Exception as exc:
        prepare_error = exc

    for benchmark in BENCHMARK_SYMBOLS:
        paths = benchmark_raw_paths(benchmark)
        try:
            frame = load_benchmark_ohlcv(benchmark)
        except Exception as exc:
            raise RuntimeError(
                f"Required benchmark OHLCV is missing or empty for {benchmark}. "
                f"Expected one of: {', '.join(paths)}"
            ) from (prepare_error or exc)
        _require_non_empty_ohlcv(frame, label=benchmark, paths=paths)

    print("runtime OHLCV precheck ok:", f"symbol={symbol_upper}", "benchmarks=SPY,^VIX")


def _check_direct(symbol: str) -> None:
    _check_runtime_ohlcv(symbol)
    artifact_dir = resolve_artifact_dir()
    loaded = load_deployed_model(artifact_dir)
    snapshot = PatternAnalysisService(cache=None, enabled=False).get_or_build(
        symbol,
        loaded,
    )
    forecast = pattern_forecast_from_prediction(
        snapshot.prediction_payload,
        symbol=symbol,
    )
    intelligence = pattern_intelligence_from_dict(snapshot.pattern_intelligence)

    if not forecast.as_of_date:
        raise RuntimeError("Pattern forecast missing as_of_date")
    if intelligence.symbol != symbol.strip().upper():
        raise RuntimeError("Pattern intelligence symbol mismatch")

    print(
        "direct pattern smoke ok:",
        f"artifact_dir={artifact_dir}",
        f"symbol={forecast.model_dump(mode='json', by_alias=True).get('symbol', symbol.upper())}",
        f"as_of={forecast.as_of_date}",
    )


def _get_json(base_url: str, path: str, token: str, symbol: str) -> dict:
    response = requests.get(
        f"{base_url.rstrip('/')}{path}",
        params={"symbol": symbol},
        headers={"Authorization": f"Bearer {token}"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def _check_http(base_url: str, token: str, symbol: str) -> None:
    pattern = _get_json(base_url, "/pattern/intelligence", token, symbol)
    if pattern.get("symbol") != symbol.strip().upper():
        raise RuntimeError("/pattern/intelligence returned the wrong symbol")
    if not pattern.get("trendContext") or not pattern.get("scores"):
        raise RuntimeError("/pattern/intelligence missing pattern payload")

    research = _get_json(base_url, "/research/intelligence", token, symbol)
    if not research.get("patternForecast"):
        raise RuntimeError("/research/intelligence missing patternForecast")
    if not research.get("patternIntelligence"):
        raise RuntimeError("/research/intelligence missing patternIntelligence")

    print("http pattern smoke ok:", f"base_url={base_url}", f"symbol={symbol.upper()}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify deployed pattern artifacts and optional paid HTTP endpoints.",
    )
    parser.add_argument("--symbol", default="NVDA")
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--access-token", default="")
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Only verify artifact loading and direct PatternAnalysisService output.",
    )
    parser.add_argument(
        "--precheck-only",
        action="store_true",
        help="Only verify required target, SPY, and ^VIX OHLCV runtime data.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    symbol = args.symbol.strip().upper()
    if args.precheck_only:
        _check_runtime_ohlcv(symbol)
        return 0

    _check_direct(symbol)

    if args.direct_only:
        return 0
    if not args.access_token:
        print(
            "Skipping authenticated endpoint checks; pass --access-token to verify "
            "/pattern/intelligence and /research/intelligence.",
        )
        return 0

    _check_http(args.base_url, args.access_token, symbol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
