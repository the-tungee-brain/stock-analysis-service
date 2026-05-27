MISC_SECTOR_LABEL = "Misc"
ETF_SECTOR_LABEL = "ETF"

ETF_INSTRUMENT_ASSET_TYPES = frozenset(
    {
        "ETF",
        "COLLECTIVE_INVESTMENT",
    }
)

_UNKNOWN_SECTOR_VALUES = frozenset(
    {
        "",
        "unknown",
        "unknown sector",
        "n/a",
        "na",
        "none",
        "uncategorized",
    }
)


def normalize_sector_label(
    sector: str | None,
    *,
    default: str = MISC_SECTOR_LABEL,
) -> str:
    if sector is None:
        return default

    cleaned = sector.strip()
    if not cleaned or cleaned.lower() in _UNKNOWN_SECTOR_VALUES:
        return default

    return cleaned


def build_asset_type_by_symbol(
    positions: list,
    *,
    research_asset_types: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map symbols to ETF when held as ETF/collective investment."""
    out: dict[str, str] = {}

    for position in positions:
        if position.instrument.assetType in ETF_INSTRUMENT_ASSET_TYPES:
            out[position.instrument.symbol.upper()] = ETF_SECTOR_LABEL

    for symbol, asset_type in (research_asset_types or {}).items():
        if asset_type in {ETF_SECTOR_LABEL, "ETF"}:
            out[symbol.upper()] = ETF_SECTOR_LABEL

    return out


def sector_label_for_holding(
    *,
    symbol: str,
    instrument_asset_type: str,
    sector_by_symbol: dict[str, str],
    asset_type_by_symbol: dict[str, str] | None = None,
) -> str:
    symbol_upper = symbol.upper()
    resolved_asset_type = (asset_type_by_symbol or {}).get(symbol_upper)
    if resolved_asset_type == ETF_SECTOR_LABEL:
        return ETF_SECTOR_LABEL
    if instrument_asset_type in ETF_INSTRUMENT_ASSET_TYPES:
        return ETF_SECTOR_LABEL
    return normalize_sector_label(sector_by_symbol.get(symbol_upper))
