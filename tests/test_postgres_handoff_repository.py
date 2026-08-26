"""Tests for PostgresHandoffRepository using fake DB connections."""

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

import app.repositories.postgres_handoff_repository as module
from app.db.connection import DatabaseNotConfiguredError
from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import HandoffReason, HandoffRequest, PersistedHandoff
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
    HandoffRepositoryDuplicateError,
)
from app.repositories.postgres_handoff_repository import PostgresHandoffRepository


def _state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
        booking_stage=BookingStage.READY_FOR_REVIEW,
        needs_human=True,
    )


def _request() -> HandoffRequest:
    return HandoffRequest(
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        conversation_state=_state(),
    )


KEY = "b" * 64


def _fake_db(row=None, execute_error=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    if execute_error is not None:
        cursor.execute.side_effect = execute_error
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=conn)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory, conn, cursor


def _repo_with(factory):
    return patch(
        "app.repositories.postgres_handoff_repository.database_connection", factory
    )


FULL_ROW = {
    "id": "12345678-1234-5678-1234-567812345678",
    "idempotency_key": KEY,
    "customer_phone": "+90555 111 2233",
    "customer_name": "Mehmet Cam",
    "reason": "booking_review",
    "status": "pending",
    "intent": "booking_request",
    "tour": "Ephesus tour",
    "travel_date": date(2026, 9, 10),
    "adults": 2,
    "children": 1,
    "cruise_ship": "Equinox",
    "hotel": "Korumar",
    "pickup_location": "Port",
    "preferred_language": "English",
    "booking_stage": "ready_for_review",
    "needs_human": True,
}


def test_repository_subclasses_handoff_repository():
    assert issubclass(PostgresHandoffRepository, HandoffRepository)
    assert isinstance(PostgresHandoffRepository(), HandoffRepository)


def test_create_uses_parameterized_insert_with_idempotency_key():
    factory, _, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().create(_request(), KEY)

    sql = cursor.execute.call_args.args[0]
    assert "INSERT INTO handoff_requests" in sql
    assert "RETURNING" in sql
    assert "%(idempotency_key)s" in sql
    assert "%s" not in sql


def test_create_generates_uuid_in_python():
    factory, _, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().create(_request(), KEY)

    assert isinstance(cursor.execute.call_args.args[1]["id"], UUID)


def test_create_inserts_snapshot_and_key_fields():
    factory, _, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().create(_request(), KEY)

    params = cursor.execute.call_args.args[1]
    expected = {
        "idempotency_key": KEY,
        "customer_phone": "+90555 111 2233",
        "customer_name": "Mehmet Cam",
        "reason": "booking_review",
        "status": "pending",
        "tour": "Ephesus tour",
        "adults": 2,
        "booking_stage": "ready_for_review",
    }
    for key_, value in expected.items():
        assert params[key_] == value


def test_create_returning_mapped_to_persisted_handoff():
    factory, _, _ = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        persisted = PostgresHandoffRepository().create(_request(), KEY)

    assert isinstance(persisted, PersistedHandoff)
    assert persisted.idempotency_key == KEY
    assert persisted.reason is HandoffReason.BOOKING_REVIEW


def test_create_commits_exactly_once():
    factory, conn, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().create(_request(), KEY)

    assert conn.commit.call_count == 1


def test_create_execute_failure_no_commit_and_propagates():
    factory, conn, cursor = _fake_db(execute_error=RuntimeError("db exploded"))
    with _repo_with(factory):
        with pytest.raises(RuntimeError, match="db exploded"):
            PostgresHandoffRepository().create(_request(), KEY)

        conn.commit.assert_not_called()


def test_duplicate_key_recovers_existing_row():
    """UniqueViolation on idempotency_key -> refetch and return existing."""
    duplicate_error = module.psycopg_errors.UniqueViolation("dup key")
    insert_factory, _, _ = _fake_db(execute_error=duplicate_error)
    repo = PostgresHandoffRepository()

    from app.repositories.handoff_mapping import db_row_to_persisted_handoff

    with _repo_with(insert_factory), patch.object(
        repo,
        "get_by_idempotency_key",
        return_value=db_row_to_persisted_handoff(dict(FULL_ROW)),
    ):
        persisted = repo.create(_request(), KEY)

    assert isinstance(persisted, PersistedHandoff)
    assert persisted.idempotency_key == KEY


def test_duplicate_key_without_recoverable_row_raises_domain_error():
    duplicate_error = module.psycopg_errors.UniqueViolation("dup key")
    insert_factory, _, _ = _fake_db(execute_error=duplicate_error)
    repo = PostgresHandoffRepository()

    with _repo_with(insert_factory), patch.object(
        repo, "get_by_idempotency_key", return_value=None
    ):
        with pytest.raises(HandoffRepositoryDuplicateError):
            repo.create(_request(), KEY)


def test_unrelated_db_errors_propagate():
    factory, _, cursor = _fake_db(row=None, execute_error=RuntimeError("boom"))
    with _repo_with(factory):
        with pytest.raises(RuntimeError, match="boom"):
            PostgresHandoffRepository().create(_request(), KEY)


def test_get_parameterized_by_uuid():
    factory, _, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().get(UUID(FULL_ROW["id"]))

    sql, params = cursor.execute.call_args.args
    assert "WHERE id = %s" in sql
    assert params == (UUID(FULL_ROW["id"]),)


def test_get_maps_row_to_persisted_handoff():
    factory, _, _ = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        persisted = PostgresHandoffRepository().get(UUID(FULL_ROW["id"]))

    assert isinstance(persisted, PersistedHandoff)
    assert persisted.conversation_state.tour == "Ephesus tour"


def test_get_missing_row_returns_none():
    factory, _, cursor = _fake_db(row=None)
    with _repo_with(factory):
        assert PostgresHandoffRepository().get(uuid4()) is None


def test_get_does_not_commit():
    factory, conn, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().get(UUID(FULL_ROW["id"]))

    conn.commit.assert_not_called()


def test_get_by_idempotency_key_parameterized():
    factory, _, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().get_by_idempotency_key(KEY)

    sql, params = cursor.execute.call_args.args
    assert "WHERE idempotency_key = %s" in sql
    assert params == (KEY,)
    assert KEY not in sql


def test_get_by_idempotency_key_missing_returns_none():
    factory, _, cursor = _fake_db(row=None)
    with _repo_with(factory):
        assert PostgresHandoffRepository().get_by_idempotency_key(KEY) is None


def test_get_by_idempotency_key_does_not_commit():
    factory, conn, cursor = _fake_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().get_by_idempotency_key(KEY)

    conn.commit.assert_not_called()


def test_no_sql_string_interpolation():
    source = module._INSERT_SQL + module._SELECT_BY_KEY_SQL + module._SELECT_SQL
    assert "{" not in source and "}" not in source


def test_no_delete_truncate_drop():
    source = (
        module._INSERT_SQL + module._SELECT_SQL + module._SELECT_BY_KEY_SQL
    ).upper()
    for forbidden in ("DELETE FROM", "TRUNCATE", "DROP", "ALTER TABLE"):
        assert forbidden not in source


def test_uses_database_connection_abstraction():
    import inspect

    repo_src = inspect.getsource(module)
    assert "database_connection()" in repo_src
    assert "psycopg.connect" not in repo_src


def test_no_openrouter_safeai_route_dependencies():
    assert not hasattr(module, "OpenRouterProvider")
    assert not hasattr(module, "SafeAIService")
    assert not hasattr(module, "router")


def test_database_not_configured_error_propagates():
    factory = MagicMock()
    factory.return_value.__enter__.side_effect = DatabaseNotConfiguredError("no db")
    factory.return_value.__exit__ = MagicMock(return_value=False)
    with _repo_with(factory):
        with pytest.raises(DatabaseNotConfiguredError):
            PostgresHandoffRepository().create(_request(), KEY)


# --- Lifecycle: update_status ----------------------------------------------------


def _update_db(row=None, execute_error=None):
    return _fake_db(row=row, execute_error=execute_error)


def test_update_status_uses_update_statement():
    from app.models.handoff import HandoffStatus

    factory, conn, cursor = _update_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().update_status(
            UUID(FULL_ROW["id"]), HandoffStatus.IN_REVIEW
        )

    sql = cursor.execute.call_args.args[0]
    assert sql.lstrip().startswith("UPDATE handoff_requests")
    assert "SET status = %s, updated_at = NOW()" in sql
    assert "WHERE id = %s" in sql
    assert "RETURNING" in sql


def test_update_status_parameterized_status_and_uuid():
    from app.models.handoff import HandoffStatus

    factory, _, cursor = _update_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().update_status(
            UUID(FULL_ROW["id"]), HandoffStatus.IN_REVIEW
        )

    params = cursor.execute.call_args.args[1]
    assert params == ("in_review", UUID(FULL_ROW["id"]))


def test_update_status_only_touches_status_and_updated_at():
    from app.models.handoff import HandoffStatus

    factory, _, cursor = _update_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().update_status(
            UUID(FULL_ROW["id"]), HandoffStatus.RESOLVED
        )

    set_clause = (
        cursor.execute.call_args.args[0].split("SET", 1)[1].split("WHERE", 1)[0]
    )
    assert "status" in set_clause
    assert "updated_at = NOW()" in set_clause
    for forbidden in (
        "reason",
        "idempotency_key",
        "customer_phone",
        "customer_name",
        "intent",
        "tour",
        "travel_date",
        "adults",
        "children",
        "booking_stage",
        "needs_human",
        "created_at",
    ):
        assert forbidden not in set_clause


def test_update_status_commits_exactly_once():
    from app.models.handoff import HandoffStatus

    factory, conn, cursor = _update_db(row=dict(FULL_ROW))
    with _repo_with(factory):
        PostgresHandoffRepository().update_status(
            UUID(FULL_ROW["id"]), HandoffStatus.IN_REVIEW
        )

    assert conn.commit.call_count == 1


def test_update_status_failure_no_commit_and_propagates():
    from app.models.handoff import HandoffStatus

    factory, conn, cursor = _update_db(execute_error=RuntimeError("db exploded"))
    with _repo_with(factory):
        with pytest.raises(RuntimeError, match="db exploded"):
            PostgresHandoffRepository().update_status(
                UUID(FULL_ROW["id"]), HandoffStatus.IN_REVIEW
            )

    conn.commit.assert_not_called()


def test_update_status_missing_row_raises_not_found():
    from app.models.handoff import HandoffStatus

    factory, conn, cursor = _update_db(row=None)
    with _repo_with(factory):
        with pytest.raises(HandoffNotFoundError):
            PostgresHandoffRepository().update_status(
                uuid4(), HandoffStatus.IN_REVIEW
            )

    conn.commit.assert_not_called()


def test_update_status_maps_returning_row():
    from app.models.handoff import HandoffStatus

    row = dict(FULL_ROW)
    row["status"] = "in_review"
    factory, _, _ = _update_db(row=row)
    with _repo_with(factory):
        persisted = PostgresHandoffRepository().update_status(
            UUID(FULL_ROW["id"]), HandoffStatus.IN_REVIEW
        )

    assert isinstance(persisted, PersistedHandoff)
    assert persisted.status is HandoffStatus.IN_REVIEW
    assert persisted.idempotency_key == KEY


def test_update_sql_has_no_destructive_or_interpolated_content():
    source = module._UPDATE_STATUS_SQL.upper()
    for forbidden in ("DELETE FROM", "TRUNCATE", "DROP", "ALTER TABLE", "{", "}"):
        assert forbidden not in source


def test_update_uses_database_connection_abstraction():
    import inspect

    repo_src = inspect.getsource(module)
    assert "database_connection()" in repo_src
    assert "psycopg.connect" not in repo_src


