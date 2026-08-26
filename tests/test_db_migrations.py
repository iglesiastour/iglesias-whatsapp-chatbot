"""SQL safety and migration runner tests (no real database)."""

from unittest.mock import MagicMock, patch

import pytest

from app.models.conversation import BookingStage, ConversationIntent
from app.models.handoff import HandoffReason, HandoffStatus
from app.db.migrations.runner import (
    MIGRATIONS,
    Migration,
    load_migration_sql,
    run_migrations,
)


def _sql() -> str:
    return load_migration_sql(MIGRATIONS[0])


# --- Registry ---


def test_migration_file_exists() -> None:
    assert load_migration_sql(MIGRATIONS[0]).strip() != ""


def test_registry_contains_migrations_0001_and_0002() -> None:
    assert len(MIGRATIONS) == 2
    assert MIGRATIONS[0].version == "0001"
    assert MIGRATIONS[0].filename == "0001_conversation_states.sql"
    assert MIGRATIONS[1].version == "0002"
    assert MIGRATIONS[1].filename == "0002_handoff_requests.sql"


def test_0002_registered_after_0001() -> None:
    versions = [m.version for m in MIGRATIONS]
    assert versions.index("0002") > versions.index("0001")


def test_registry_order_deterministic() -> None:
    versions = [m.version for m in MIGRATIONS]
    assert versions == sorted(versions)


# --- SQL content ---


def test_sql_contains_create_table_if_not_exists() -> None:
    assert "CREATE TABLE IF NOT EXISTS conversation_states" in _sql()


def test_customer_phone_is_primary_key() -> None:
    assert "customer_phone TEXT PRIMARY KEY" in _sql()


def test_required_intent_column_exists() -> None:
    assert "intent TEXT NOT NULL" in _sql()


def test_booking_stage_column_exists() -> None:
    assert "booking_stage TEXT NOT NULL" in _sql()


def test_needs_human_column_exists() -> None:
    assert "needs_human BOOLEAN NOT NULL DEFAULT FALSE" in _sql()


def test_travel_date_uses_date_type() -> None:
    assert "travel_date DATE NULL" in _sql()


def test_created_at_and_updated_at_exist() -> None:
    sql = _sql()
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in sql
    assert "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in sql


def test_adults_check_constraint_exists() -> None:
    assert "adults IS NULL OR (adults >= 1 AND adults <= 100)" in _sql()


def test_children_check_constraint_exists() -> None:
    assert "children IS NULL OR (children >= 0 AND children <= 100)" in _sql()


def test_all_conversation_intent_values_in_sql() -> None:
    for intent in ConversationIntent:
        assert f"'{intent.value}'" in _sql()


def test_all_booking_stage_values_in_sql() -> None:
    for stage in BookingStage:
        assert f"'{stage.value}'" in _sql()


# --- Non-destructive / safety ---


@pytest.mark.parametrize(
    "forbidden",
    ["DROP TABLE", "TRUNCATE", "DELETE FROM", "ALTER TABLE", "DROP CONSTRAINT"],
)
def test_no_destructive_sql(forbidden: str) -> None:
    assert forbidden.upper() not in _sql().upper()


def test_no_secrets_in_sql() -> None:
    sql = _sql().lower()
    for forbidden in ("api_key", "access_token", "secret", "password", "bearer"):
        assert forbidden not in sql


# --- Loader safety ---


def test_unknown_migration_rejected() -> None:
    rogue = Migration(version="9999", filename="9999_rogue.sql")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_migration_sql(rogue)


def test_missing_file_raises_clear_error(tmp_path, monkeypatch) -> None:
    from app.db.migrations import runner

    missing = Migration(version="0002", filename="0002_missing.sql")
    monkeypatch.setattr(runner, "_MIGRATIONS_DIR", tmp_path)

    with pytest.raises(FileNotFoundError):
        runner.load_migration_sql(missing)


# --- Runner ---


def _fake_conn():
    conn = MagicMock()
    cursor_ctx = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor_ctx)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return conn, conn.cursor.return_value.__enter__.return_value


def test_import_does_not_connect(monkeypatch) -> None:
    from app.config import settings
    from app.db.migrations import runner

    fake_connect = MagicMock()
    monkeypatch.setattr(settings, "database_url", "")
    with patch.object(runner, "database_connection") as factory:
        for name in dir(runner):
            if not name.startswith("_"):
                getattr(runner, name)
    assert factory.call_count == 0
    assert fake_connect.call_count == 0


def test_runner_execution_exception_propagates(monkeypatch) -> None:
    from app.config import settings
    from app.db.migrations import runner

    monkeypatch.setattr(settings, "database_url", "postgresql://user@host/db")
    conn, cursor = _fake_conn()
    cursor.execute.side_effect = RuntimeError("db exploded")

    with patch.object(runner, "database_connection") as factory:
        factory.return_value.__enter__ = MagicMock(return_value=conn)
        factory.return_value.__exit__ = MagicMock(return_value=False)
        with pytest.raises(RuntimeError, match="db exploded"):
            run_migrations()




# --- 0002: handoff_requests schema ---


def _sql_0002() -> str:
    return load_migration_sql(MIGRATIONS[1])


def test_0002_creates_handoff_requests_table() -> None:
    assert "CREATE TABLE IF NOT EXISTS handoff_requests" in _sql_0002()


def test_0002_id_is_uuid_primary_key() -> None:
    assert "id UUID PRIMARY KEY" in _sql_0002()


def test_0002_required_columns_exist() -> None:
    sql = _sql_0002()
    for column in (
        "customer_phone TEXT NOT NULL",
        "customer_name TEXT NULL",
        "reason TEXT NOT NULL",
        "status TEXT NOT NULL",
        "intent TEXT NOT NULL",
        "tour TEXT NULL",
        "travel_date DATE NULL",
        "adults INTEGER NULL",
        "children INTEGER NULL",
        "cruise_ship TEXT NULL",
        "hotel TEXT NULL",
        "pickup_location TEXT NULL",
        "preferred_language TEXT NULL",
        "booking_stage TEXT NOT NULL",
        "needs_human BOOLEAN NOT NULL",
    ):
        assert column in sql


def test_0002_created_at_and_updated_at_exist() -> None:
    sql = _sql_0002()
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in sql
    assert "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in sql


def test_0002_all_handoff_reason_values_in_check() -> None:
    sql = _sql_0002()
    for reason in HandoffReason:
        assert f"'{reason.value}'" in sql


def test_0002_all_handoff_status_values_in_check() -> None:
    sql = _sql_0002()
    for status in HandoffStatus:
        assert f"'{status.value}'" in sql


def test_0002_all_conversation_intent_values_in_check() -> None:
    sql = _sql_0002()
    for intent in ConversationIntent:
        assert f"'{intent.value}'" in sql


def test_0002_all_booking_stage_values_in_check() -> None:
    sql = _sql_0002()
    for stage in BookingStage:
        assert f"'{stage.value}'" in sql


def test_0002_numeric_constraints_present() -> None:
    sql = _sql_0002()
    assert "adults IS NULL OR (adults >= 1 AND adults <= 100)" in sql
    assert "children IS NULL OR (children >= 0 AND children <= 100)" in sql


@pytest.mark.parametrize(
    "forbidden",
    ["DROP TABLE", "TRUNCATE", "DELETE FROM", "ALTER TABLE", "DROP CONSTRAINT"],
)
def test_0002_no_destructive_sql(forbidden: str) -> None:
    assert forbidden.upper() not in _sql_0002().upper()


def test_0002_idempotency_key_column_exists_not_null_unique() -> None:
    sql = _sql_0002()
    assert "idempotency_key TEXT NOT NULL UNIQUE" in sql


def test_0002_no_secret_like_literals() -> None:
    sql = _sql_0002().lower()
    for forbidden in ("api_key", "access_token", "secret", "password", "bearer"):
        assert forbidden not in sql

