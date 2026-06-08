from pathlib import Path


MIGRATION_SQL = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "sql"
    / "migrations"
    / "20260605_provider_symbol_profile.sql"
)


def _normalized_sql() -> str:
    return MIGRATION_SQL.read_text(encoding="utf-8").lower()


def test_provider_symbol_profile_migration_is_idempotent_and_additive():
    sql = _normalized_sql()

    assert "user_tables" in sql
    assert "user_tab_columns" in sql
    assert "provider_symbol_profile" in sql
    assert "table_count = 0" in sql
    assert "execute immediate" in sql
    assert "create table provider_symbol_profile" in sql
    assert "alter table provider_symbol_profile add" in sql
    assert "constraint provider_symbol_profile_pk primary key (provider, symbol)" in sql


def test_provider_symbol_profile_migration_stores_public_provider_metadata_only():
    sql = _normalized_sql()

    expected_columns = [
        "provider",
        "symbol",
        "status",
        "fetched_at",
        "updated_at",
        "dividend_yield_pct",
        "raw_dividend_yield",
        "raw_dividend_yield_source",
        "raw_json",
    ]
    for column in expected_columns:
        assert column in sql

    private_terms = [
        "user_id",
        "account",
        "access_token",
        "refresh_token",
        "portfolio",
        "prompt",
    ]
    for term in private_terms:
        assert term not in sql


def test_provider_symbol_profile_migration_does_not_touch_ticker_symbols():
    sql = _normalized_sql()

    destructive_or_identity_changes = [
        "drop table",
        "truncate",
        "delete from ticker_symbols",
        "alter table ticker_symbols",
        "create table ticker_symbols",
    ]
    for phrase in destructive_or_identity_changes:
        assert phrase not in sql
