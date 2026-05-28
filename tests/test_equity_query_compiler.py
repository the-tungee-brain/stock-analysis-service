from app.models.screener_preset_models import EquityQueryClause, EquityQuerySpec
from app.screener.equity_query_compiler import compile_equity_query, resolve_field, resolve_sector


def test_field_aliases_map_to_yfinance_names():
    assert resolve_field("marketCap") == "intradaymarketcap"
    assert resolve_field("regularMarketPrice") == "intradayprice"
    assert resolve_field("trailingPE") == "peratio.lasttwelvemonths"
    assert resolve_field("dividendYield") == "forward_dividend_yield"


def test_sector_aliases_map_consumer_staples():
    assert resolve_sector("Consumer Staples") == "Consumer Defensive"


def test_compile_is_in_sector_clause():
    spec = EquityQuerySpec(
        operator="and",
        clauses=[
            EquityQueryClause(op="eq", field="region", value="us"),
            EquityQueryClause(
                op="is-in",
                field="sector",
                values=["Technology", "Consumer Staples"],
            ),
        ],
    )
    query = compile_equity_query(spec)
    sector_clause = query.operands[1]
    assert sector_clause.operands[0] == "sector"
    assert "Consumer Defensive" in sector_clause.operands
