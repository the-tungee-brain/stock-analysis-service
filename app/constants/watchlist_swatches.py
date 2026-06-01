"""Folder background swatch IDs — keep in sync with iOS `WatchlistPremiumPalette`."""

DEFAULT_WATCHLIST_SWATCH_ID = "slate"

# Classic tones + premium abstract backgrounds (Tomcrest iOS folder picker).
WATCHLIST_SWATCH_IDS: frozenset[str] = frozenset(
    {
        # Classic
        "mauve",
        "sage",
        "lavender",
        "teal",
        "sand",
        "rose",
        "slate",
        "ocean",
        # Orbs & light
        "orb-iris",
        "orb-mist",
        "twin-rose",
        "twin-jade",
        "sphere-noir",
        "radial-dawn",
        "halo-moon",
        # Lines & flow
        "flow-azure",
        "flow-plum",
        "ribbon-sand",
        "ribbon-noir",
        "orbital-gold",
        "orbital-steel",
        "arcs-minimal",
        "liquid-teal",
        "streak-ice",
        "wave-midnight",
        "metal-champagne",
        # Glass & depth
        "glass-pearl",
        "glass-obsidian",
        "layer-sage",
        "layer-indigo",
        "stack-glass",
        "block-graphite",
        "arch-stone",
        "vision-aqua",
        "vision-blush",
        # Pattern & form
        "ring-slate",
        "ring-copper",
        "grid-fog",
        "blob-emerald",
        "blob-lilac",
        "mesh-coral",
        "particle-noir",
    }
)


def normalize_watchlist_swatch_id(swatch_id: str | None) -> str:
    if swatch_id and swatch_id in WATCHLIST_SWATCH_IDS:
        return swatch_id
    return DEFAULT_WATCHLIST_SWATCH_ID
