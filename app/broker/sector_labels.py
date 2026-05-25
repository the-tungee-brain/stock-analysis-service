MISC_SECTOR_LABEL = "Misc"

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
