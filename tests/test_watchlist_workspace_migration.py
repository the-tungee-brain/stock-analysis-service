from pathlib import Path


MIGRATION_SQL = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "sql"
    / "migrations"
    / "20260604_watchlist_workspace.sql"
)


def _normalized_sql() -> str:
    return MIGRATION_SQL.read_text(encoding="utf-8").lower()


def test_watchlist_workspace_migration_is_idempotent_create_only():
    sql = _normalized_sql()

    assert "user_tables" in sql
    assert "watchlist_workspace" in sql
    assert "table_count = 0" in sql
    assert "execute immediate" in sql
    assert "create table watchlist_workspace" in sql


def test_watchlist_workspace_migration_does_not_touch_existing_watchlist_data():
    sql = _normalized_sql()

    destructive_phrases = [
        "drop table",
        "truncate",
        "delete from watchlist_folder",
        "delete from watchlist_item",
        "alter table watchlist_folder",
        "alter table watchlist_item",
        "create table watchlist_folder",
        "create table watchlist_item",
    ]

    for phrase in destructive_phrases:
        assert phrase not in sql
