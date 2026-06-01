"""Static symbol-to-sector mapping for portfolio concentration reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from data.paths import PROJECT_ROOT

SECTOR_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "sector_mapping.json"
UNKNOWN_SECTOR = "Unknown"

# Reproducible sector labels for production universes (Yahoo-style sector names).
_STATIC_SECTORS: dict[str, str] = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "AMZN": "Consumer Cyclical",
    "META": "Communication Services",
    "GOOGL": "Communication Services",
    "NVDA": "Technology",
    "TSLA": "Consumer Cyclical",
    "BRK-B": "Financial Services",
    "JPM": "Financial Services",
    "V": "Financial Services",
    "JNJ": "Healthcare",
    "HD": "Consumer Cyclical",
    "PG": "Consumer Defensive",
    "MA": "Financial Services",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "SPY": "Broad Market",
    "LLY": "Healthcare",
    "AVGO": "Technology",
    "COST": "Consumer Defensive",
    "ORCL": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "NFLX": "Communication Services",
    "PEP": "Consumer Defensive",
    "KO": "Consumer Defensive",
    "WMT": "Consumer Defensive",
    "DIS": "Communication Services",
    "BAC": "Financial Services",
    "GS": "Financial Services",
    "MRK": "Healthcare",
    "ABBV": "Healthcare",
    "TMO": "Healthcare",
    "CSCO": "Technology",
    "ACN": "Technology",
    "LIN": "Basic Materials",
    "NKE": "Consumer Cyclical",
    "PM": "Consumer Defensive",
    "MCD": "Consumer Cyclical",
    "WFC": "Financial Services",
    "IBM": "Technology",
    "TXN": "Technology",
    "QCOM": "Technology",
    "AMAT": "Technology",
    "INTU": "Technology",
    "ISRG": "Healthcare",
    "BKNG": "Consumer Cyclical",
    "AMD": "Technology",
    "CAT": "Industrials",
    "DE": "Industrials",
    "UPS": "Industrials",
    "BA": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "LOW": "Consumer Cyclical",
    "SBUX": "Consumer Cyclical",
    "TJX": "Consumer Cyclical",
    "MDLZ": "Consumer Defensive",
    "GILD": "Healthcare",
    "REGN": "Healthcare",
    "CI": "Healthcare",
    "ELV": "Healthcare",
    "MO": "Consumer Defensive",
    "NEE": "Utilities",
    "MMM": "Industrials",
    "AXP": "Financial Services",
    "BLK": "Financial Services",
    "SPGI": "Financial Services",
    "CME": "Financial Services",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "GM": "Consumer Cyclical",
    "F": "Consumer Cyclical",
    "MU": "Technology",
    "LRCX": "Technology",
    "KLAC": "Technology",
    "PANW": "Technology",
    "CRWD": "Technology",
    "UBER": "Consumer Cyclical",
    "ABNB": "Consumer Cyclical",
    "PYPL": "Financial Services",
    "ADP": "Technology",
    "SYK": "Healthcare",
    "ZTS": "Healthcare",
    "CB": "Financial Services",
    "PGR": "Financial Services",
    "SO": "Utilities",
    "DUK": "Utilities",
    "CL": "Consumer Defensive",
    "BMY": "Healthcare",
    "SCHW": "Financial Services",
    "ETN": "Industrials",
    "ITW": "Industrials",
    "HCA": "Healthcare",
    "TMUS": "Communication Services",
    "SHW": "Basic Materials",
    "ICE": "Financial Services",
    "PNC": "Financial Services",
}


def load_sector_mapping(*, refresh: bool = False) -> dict[str, str]:
    """Return symbol→sector map, optionally loading cached JSON overrides."""
    mapping = {symbol.upper(): sector for symbol, sector in _STATIC_SECTORS.items()}
    if refresh or not SECTOR_CACHE_PATH.exists():
        return mapping

    try:
        cached = json.loads(SECTOR_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(cached, dict):
            for symbol, sector in cached.items():
                mapping[str(symbol).upper()] = str(sector)
    except (OSError, json.JSONDecodeError):
        pass
    return mapping


def get_sector(symbol: str, mapping: Mapping[str, str] | None = None) -> str:
    """Look up sector for one symbol."""
    sectors = mapping or load_sector_mapping()
    return sectors.get(symbol.strip().upper(), UNKNOWN_SECTOR)


def attach_sector_column(
    frame: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    mapping: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Add ``sector`` column to a predictions or panel frame."""
    sectors = mapping or load_sector_mapping()
    out = frame.copy()
    out["sector"] = out[symbol_col].astype(str).str.upper().map(lambda s: sectors.get(s, UNKNOWN_SECTOR))
    return out


def sector_map_for_symbols(symbols: Sequence[str]) -> dict[str, str]:
    """Return sector mapping restricted to requested symbols."""
    sectors = load_sector_mapping()
    return {symbol.upper(): get_sector(symbol, sectors) for symbol in symbols}


def save_sector_cache(mapping: Mapping[str, str]) -> Path:
    """Persist sector overrides to JSON cache."""
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(k).upper(): str(v) for k, v in mapping.items()}
    SECTOR_CACHE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return SECTOR_CACHE_PATH
