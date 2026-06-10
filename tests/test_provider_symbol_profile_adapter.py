import math

from app.adapters.market.provider_symbol_profile_adapter import (
    ProviderSymbolProfileAdapter,
)


class _FakeCursor:
    def __init__(
        self,
        *,
        table_exists: bool = True,
        existing_columns: set[str] | None = None,
    ) -> None:
        self.table_exists = table_exists
        self.existing_columns = existing_columns or set()
        self.statements: list[str] = []
        self._last_result = None

    def execute(self, sql: str, params=None):
        self.statements.append(" ".join(sql.split()))
        normalized = sql.lower()
        params = params or {}
        if "from user_tables" in normalized:
            self._last_result = (1,) if self.table_exists else None
            return
        if "from user_tab_columns" in normalized:
            column = str(params.get("column_name", "")).upper()
            self._last_result = (1,) if column in self.existing_columns else None
            return
        if normalized.strip().startswith("alter table"):
            start = normalized.index("add (") + len("add (")
            column = normalized[start:].split()[0].upper()
            self.existing_columns.add(column)
            self._last_result = None
            return
        if normalized.strip().startswith("create table"):
            self.table_exists = True
            self.existing_columns.update(
                column for column, _ in ProviderSymbolProfileAdapter._required_additive_columns()
            )
            self._last_result = None
            return
        self._last_result = None

    def fetchone(self):
        return self._last_result


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.cursor_obj = cursor
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


class _FakePool:
    def __init__(self, cursor: _FakeCursor) -> None:
        self.connection = _FakeConnection(cursor)
        self.released = False

    def acquire(self):
        return self.connection

    def release(self, _connection):
        self.released = True


def test_provider_symbol_profile_adapter_adds_missing_dividend_columns():
    cursor = _FakeCursor(
        table_exists=True,
        existing_columns={"RAW_DIVIDEND_YIELD"},
    )
    pool = _FakePool(cursor)

    ProviderSymbolProfileAdapter(pool)  # type: ignore[arg-type]

    alter_statements = [
        statement for statement in cursor.statements if statement.lower().startswith("alter table")
    ]
    assert any("dividend_yield_pct" in statement.lower() for statement in alter_statements)
    assert any("raw_dividend_yield_source" in statement.lower() for statement in alter_statements)
    assert not any(
        "raw_dividend_yield number" in statement.lower()
        for statement in alter_statements
    )
    assert pool.connection.committed is True
    assert pool.released is True


def test_provider_symbol_profile_adapter_schema_is_noop_when_columns_exist():
    cursor = _FakeCursor(
        table_exists=True,
        existing_columns={
            "DIVIDEND_YIELD_PCT",
            "RAW_DIVIDEND_YIELD",
            "RAW_DIVIDEND_YIELD_SOURCE",
        },
    )

    ProviderSymbolProfileAdapter(_FakePool(cursor))  # type: ignore[arg-type]

    assert not any(
        statement.lower().startswith("alter table")
        for statement in cursor.statements
    )


def test_provider_symbol_profile_adapter_creates_table_when_missing():
    cursor = _FakeCursor(table_exists=False)

    ProviderSymbolProfileAdapter(_FakePool(cursor))  # type: ignore[arg-type]

    assert any(
        statement.lower().startswith("create table provider_symbol_profile")
        for statement in cursor.statements
    )


def test_provider_symbol_profile_adapter_treats_existing_ddl_errors_as_applied():
    class _Error:
        code = 1430

    exc = Exception(_Error())

    assert ProviderSymbolProfileAdapter._is_ddl_already_applied(exc) is True  # type: ignore[arg-type]


def test_normalized_fields_drop_non_finite_numbers():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "currentPrice": float("inf"),
            "regularMarketPreviousClose": "-inf",
            "marketCap": float("nan"),
            "totalAssets": math.inf,
            "volume": "-Infinity",
            "averageVolume": "NaN",
            "trailingPE": "-Infinity",
            "forwardPE": "Infinity",
            "priceToBook": math.nan,
            "dividendYield": float("inf"),
            "dividendRate": "-inf",
            "annualReportExpenseRatio": "nan",
            "beta": math.inf,
        }
    )

    assert fields["current_price"] is None
    assert fields["previous_close"] is None
    assert fields["market_cap"] is None
    assert fields["total_assets"] is None
    assert fields["volume"] is None
    assert fields["avg_volume"] is None
    assert fields["trailing_pe"] is None
    assert fields["forward_pe"] is None
    assert fields["price_to_book"] is None
    assert fields["dividend_yield"] is None
    assert fields["dividend_yield_pct"] is None
    assert fields["raw_dividend_yield"] is None
    assert fields["raw_dividend_yield_source"] is None
    assert fields["dividend_rate"] is None
    assert fields["expense_ratio"] is None
    assert fields["beta"] is None


def test_normalized_fields_store_raw_and_percent_point_dividend_yield():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "quoteType": "EQUITY",
            "dividendYield": 0.35,
        }
    )

    assert fields["dividend_yield"] == 0.35
    assert fields["raw_dividend_yield"] == 0.35
    assert fields["raw_dividend_yield_source"] == "yfinance.info.dividendYield"
    assert fields["dividend_yield_pct"] == 0.35


def test_normalized_fields_convert_etf_decimal_ratio_dividend_yield():
    adapter = ProviderSymbolProfileAdapter.__new__(ProviderSymbolProfileAdapter)

    fields = adapter._normalized_fields(
        {
            "quoteType": "ETF",
            "dividendYield": 0.0035,
        }
    )

    assert fields["dividend_yield"] == 0.0035
    assert fields["dividend_yield_pct"] == 0.35
