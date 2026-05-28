from __future__ import annotations

from yfinance import EquityQuery

from app.models.screener_preset_models import EquityQueryClause, EquityQuerySpec

# Preset field names -> Yahoo Finance EquityQuery field names (yfinance 1.2.0).
FIELD_ALIASES: dict[str, str] = {
    "marketCap": "intradaymarketcap",
    "regularMarketPrice": "intradayprice",
    "trailingPE": "peratio.lasttwelvemonths",
    "dividendYield": "forward_dividend_yield",
}

# Legacy / common sector labels -> Yahoo valid sector names.
SECTOR_ALIASES: dict[str, str] = {
    "Consumer Staples": "Consumer Defensive",
}


def resolve_field(field: str) -> str:
    return FIELD_ALIASES.get(field, field)


def resolve_sector(value: str) -> str:
    return SECTOR_ALIASES.get(value, value)


def compile_equity_query(spec: EquityQuerySpec) -> EquityQuery:
    clauses = [_compile_clause(clause) for clause in spec.clauses]
    return EquityQuery(spec.operator, clauses)


def _compile_clause(clause: EquityQueryClause) -> EquityQuery:
    field = resolve_field(clause.field)

    if clause.op == "is-in":
        if not clause.values:
            raise ValueError(f"is-in clause for {clause.field} requires values")
        values = clause.values
        if clause.field == "sector":
            values = [resolve_sector(str(value)) for value in values]
        return EquityQuery("is-in", [field, *values])

    if clause.op == "eq":
        if clause.value is None:
            raise ValueError(f"eq clause for {clause.field} requires value")
        value = resolve_sector(str(clause.value)) if clause.field == "sector" else clause.value
        return EquityQuery("eq", [field, value])

    if clause.op in {"gt", "gte", "lt", "lte"}:
        if clause.value is None:
            raise ValueError(f"{clause.op} clause for {clause.field} requires value")
        if not isinstance(clause.value, (int, float)):
            raise TypeError(f"{clause.op} clause for {clause.field} requires numeric value")
        return EquityQuery(clause.op, [field, clause.value])

    if clause.op == "btwn":
        if not clause.values or len(clause.values) != 2:
            raise ValueError(f"btwn clause for {clause.field} requires two numeric values")
        low, high = clause.values
        return EquityQuery("btwn", [field, low, high])

    raise ValueError(f"Unsupported equity clause op: {clause.op}")
