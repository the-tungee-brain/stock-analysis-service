from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.models.screener_preset_models import ScreenerPreset, ScreenerPresetSummary
from app.models.strategy_models import InvestmentStrategy

_PRESETS_DIR = Path(__file__).resolve().parent / "presets"

STRATEGY_PRESET_IDS: dict[InvestmentStrategy, str] = {
    InvestmentStrategy.WHEEL: "wheel_stock",
    InvestmentStrategy.CSP_INCOME: "csp_stock",
    InvestmentStrategy.COVERED_CALL: "covered_call_stock",
    InvestmentStrategy.DIVIDEND: "dividend_stock",
    InvestmentStrategy.ETF_CORE: "core_etf",
}

STRATEGY_COMPANION_PRESET_IDS: dict[InvestmentStrategy, list[str]] = {
    InvestmentStrategy.WHEEL: ["wheel_etf"],
    InvestmentStrategy.CSP_INCOME: ["csp_etf"],
    InvestmentStrategy.COVERED_CALL: ["covered_call_etf"],
    InvestmentStrategy.DIVIDEND: ["dividend_etf"],
}


def presets_for_strategy(strategy: InvestmentStrategy) -> list[str]:
    preset_ids = [STRATEGY_PRESET_IDS[strategy], *STRATEGY_COMPANION_PRESET_IDS.get(strategy, [])]
    return preset_ids


@lru_cache(maxsize=1)
def load_all_presets() -> dict[str, ScreenerPreset]:
    presets: dict[str, ScreenerPreset] = {}
    for path in sorted(_PRESETS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        preset = ScreenerPreset.model_validate(payload)
        presets[preset.id] = preset
    return presets


def get_preset(preset_id: str) -> ScreenerPreset | None:
    return load_all_presets().get(preset_id)


def preset_for_strategy(strategy: InvestmentStrategy) -> ScreenerPreset:
    preset_id = STRATEGY_PRESET_IDS[strategy]
    preset = get_preset(preset_id)
    if preset is None:
        raise KeyError(f"No screener preset registered for strategy {strategy.value}")
    return preset


def preset_summary(preset: ScreenerPreset) -> ScreenerPresetSummary:
    return ScreenerPresetSummary(
        id=preset.id,
        label=preset.label,
        description=preset.description,
        post_filters=preset.post_filters,
        post_filter_status="metadata_only",
    )


def list_preset_summaries() -> list[ScreenerPresetSummary]:
    return [preset_summary(preset) for preset in load_all_presets().values()]
