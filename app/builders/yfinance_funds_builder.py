from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from app.adapters.market.yfinance_adapter import YFinanceAdapter
from app.models.yfinance_funds_models import (
    EtfFundsSnapshot,
    FundTopHolding,
    FundWeighting,
)

logger = logging.getLogger(__name__)

_ASSET_CLASS_LABELS = {
    "cashposition": "Cash",
    "stockposition": "Stocks",
    "bondposition": "Bonds",
    "preferredposition": "Preferred",
    "convertibleposition": "Convertible",
    "otherposition": "Other",
}

_SECTOR_LABEL_OVERRIDES = {
    "realestate": "Real estate",
    "basicmaterials": "Basic materials",
    "communicationservices": "Communication services",
    "consumerCyclical": "Consumer cyclical",
}


class YFinanceFundsBuilder:
    def __init__(self, yfinance_adapter: YFinanceAdapter):
        self.yfinance_adapter = yfinance_adapter

    def build(self, symbol: str) -> EtfFundsSnapshot | None:
        symbol_upper = symbol.strip().upper()
        if not symbol_upper:
            return None

        try:
            raw = self.yfinance_adapter.get_funds_data_raw(symbol_upper)
        except Exception:
            logger.exception("yfinance funds data failed for %s", symbol_upper)
            return None

        if not raw:
            return None

        overview = raw.get("fund_overview") if isinstance(raw.get("fund_overview"), dict) else {}
        operations = self._parse_fund_operations(
            raw.get("fund_operations"), symbol_upper
        )
        asset_classes = self._parse_weighting_dict(raw.get("asset_classes"), _ASSET_CLASS_LABELS)
        sector_weightings = self._parse_weighting_dict(
            raw.get("sector_weightings"), label_formatter=_format_sector_label
        )
        bond_ratings = self._parse_weighting_dict(raw.get("bond_ratings"))
        top_holdings = self._parse_top_holdings(raw.get("top_holdings"))

        description = raw.get("description")
        if isinstance(description, str):
            description = description.strip() or None
            if description and len(description) > 480:
                description = description[:477].rstrip() + "…"

        snapshot = EtfFundsSnapshot(
            category=_optional_str(overview.get("categoryName")),
            family=_optional_str(overview.get("family")),
            legal_type=_optional_str(overview.get("legalType")),
            description=description,
            expense_ratio_pct=operations.get("expense_ratio_pct"),
            category_expense_ratio_pct=operations.get("category_expense_ratio_pct"),
            holdings_turnover_pct=operations.get("holdings_turnover_pct"),
            total_net_assets=operations.get("total_net_assets"),
            asset_classes=asset_classes,
            sector_weightings=sector_weightings,
            bond_ratings=bond_ratings,
            top_holdings=top_holdings,
        )

        if not self._has_content(snapshot):
            return None
        return snapshot

    @staticmethod
    def _has_content(snapshot: EtfFundsSnapshot) -> bool:
        return bool(
            snapshot.category
            or snapshot.family
            or snapshot.description
            or snapshot.expense_ratio_pct is not None
            or snapshot.asset_classes
            or snapshot.sector_weightings
            or snapshot.bond_ratings
            or snapshot.top_holdings
        )

    def _parse_fund_operations(
        self, df: Any, symbol: str
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {
            "expense_ratio_pct": None,
            "category_expense_ratio_pct": None,
            "holdings_turnover_pct": None,
            "total_net_assets": None,
        }
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return result

        symbol_col = symbol if symbol in df.columns else None
        if symbol_col is None:
            for col in df.columns:
                if str(col).lower() != "category average":
                    symbol_col = col
                    break

        category_col = None
        for col in df.columns:
            if "category" in str(col).lower():
                category_col = col
                break

        for idx, row in df.iterrows():
            label = str(idx).lower()
            fund_value = (
                self._normalize_pct(row[symbol_col])
                if symbol_col and symbol_col in row.index
                else None
            )
            category_value = (
                self._normalize_pct(row[category_col])
                if category_col and category_col in row.index
                else None
            )
            if "expense" in label:
                result["expense_ratio_pct"] = fund_value
                result["category_expense_ratio_pct"] = category_value
            elif "turnover" in label:
                result["holdings_turnover_pct"] = fund_value
            elif "net assets" in label:
                parsed = self._optional_float(row[symbol_col]) if symbol_col else None
                result["total_net_assets"] = parsed
        return result

    def _parse_weighting_dict(
        self,
        raw: Any,
        label_map: dict[str, str] | None = None,
        *,
        label_formatter: Any = None,
    ) -> list[FundWeighting]:
        if not isinstance(raw, dict) or not raw:
            return []

        rows: list[FundWeighting] = []
        for key, value in raw.items():
            weight = self._normalize_pct(value)
            if weight is None or weight <= 0:
                continue
            label_key = str(key).replace(" ", "")
            if label_map:
                label = label_map.get(label_key.lower()) or label_map.get(str(key).lower())
            else:
                label = None
            if not label and label_formatter:
                label = label_formatter(str(key))
            if not label:
                label = _humanize_key(str(key))
            rows.append(FundWeighting(label=label, weight_pct=weight))

        rows.sort(key=lambda row: row.weight_pct, reverse=True)
        return rows

    def _parse_top_holdings(self, df: Any, *, limit: int = 10) -> list[FundTopHolding]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        name_col = self._find_column(df, "Name", "holdingName")
        pct_col = self._find_column(df, "Holding Percent", "holdingPercent", "weight")

        holdings: list[FundTopHolding] = []
        for symbol_idx, row in df.head(limit).iterrows():
            name = (
                str(row[name_col]).strip()
                if name_col and name_col in row.index and row[name_col] is not None
                else str(symbol_idx).strip()
            )
            weight = (
                self._normalize_pct(row[pct_col])
                if pct_col and pct_col in row.index
                else None
            )
            if not name or weight is None:
                continue
            symbol = str(symbol_idx).strip().upper()
            if symbol in ("NAN", ""):
                symbol = None
            holdings.append(
                FundTopHolding(symbol=symbol, name=name, weight_pct=weight)
            )
        return holdings

    @staticmethod
    def _find_column(df: pd.DataFrame, *candidates: str) -> str | None:
        for name in candidates:
            for col in df.columns:
                if str(col).lower().replace(" ", "") == name.lower().replace(" ", ""):
                    return col
        return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if pd.isna(parsed):
            return None
        return parsed

    @staticmethod
    def _normalize_pct(value: Any) -> float | None:
        parsed = YFinanceFundsBuilder._optional_float(value)
        if parsed is None:
            return None
        if abs(parsed) <= 1.5:
            return parsed * 100
        return parsed


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _humanize_key(key: str) -> str:
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)
    spaced = spaced.replace("_", " ").strip()
    return spaced[:1].upper() + spaced[1:] if spaced else key


def _format_sector_label(key: str) -> str:
    normalized = key.replace("_", "").lower()
    if normalized in _SECTOR_LABEL_OVERRIDES:
        return _SECTOR_LABEL_OVERRIDES[normalized]
    return _humanize_key(key)
