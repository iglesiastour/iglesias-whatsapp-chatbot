"""Tests for handoff audit repositories (postgres fakes + in-memory)."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest

import app.repositories.postgres_handoff_audit_repository as pg_module
from app.models.handoff import HandoffStatus
from app.models.handoff_audit import HandoffAuditAction, HandoffAuditEvent
from app.repositories.handoff_audit_repository import HandoffAuditRepository
from app.repositories.in_memory_handoff_audit_repository import (
    InMemoryHandoffAuditRepository,
)
from app.repositories.postgres_handoff_audit_repository import (
    PostgresHandoffAuditRepository,
)

HANDOFF_ID = UUID("12345678123456781234567812345678")


def _audit_row(handoff_id=HANDOFF_ID, previous="pending", new="in_review"):
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "handoff_id": str(handoff_id),
        "action": "status_changed",
        "previous_status": previous,
        "new_status": new,
        "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
    }


def _fake_db(row=None, rows=None, execute_error=None):
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    cursor.fetchall.return_value = rows if rows is not None else []
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
    return patch.object(pg_module, "database_connection", factory)


def test_repositories_subclass_audit_repository():
    assert issubclass(PostgresHandoffAuditRepository, HandoffAuditRepository)
    assert issubclass(InMemoryHandoffAuditRepository, HandoffAuditRepository)


# --- Postgres create ---------------------------------------------------------------


def test_postgres_create_parameterized_insert():
    factory, conn, cursor = _fake_db(row=_audit_row())
    with _repo_with(factory):
        PostgresHandoffAuditRepository().create_status_change(
            handoff_id=HANDOFF_ID,
            previous_status=HandoffStatus.PENDING,
            new_status=HandoffStatus.IN_REVIEW,
        )

    sql, params = cursor.execute.call_args.args
    assert "INSERT INTO handoff_audit_events" in sql
    assert "RETURNING" in sql
    assert params["action"] == "status_changed"
    assert params["previous_status"] == "pending"
    assert params["new_status"] == "in_review"
    assert params["handoff_id"] == HANDOFF_ID
    from uuid import UUID as _UUID

    assert isinstance(params["id"], _UUID)
    conn.commit.assert_called_once()


def test_postgres_create_commits_exactly_once():
    factory, conn, cursor = _fake_db(row=_audit_row())
    with _repo_with(factory):
        PostgresHandoffAuditRepository().create_status_change(
            handoff_id=HANDOFF_ID,
            previous_status=HandoffStatus.PENDING,
            new_status=HandoffStatus.IN_REVIEW,
        )

    assert conn.commit.call_count == 1


def test_postgres_create_failure_no_commit():
    factory, conn, cursor = _fake_db(execute_error=RuntimeError("boom"))
    with _repo_with(factory):
        with pytest.raises(RuntimeError, match="boom"):
            PostgresHandoffAuditRepository().create_status_change(
                handoff_id=HANDOFF_ID,
                previous_status=HandoffStatus.PENDING,
                new_status=HandoffStatus.IN_REVIEW,
            )

    conn.commit.assert_not_called()


def test_postgres_create_maps_returning_row():
    factory, _, _ = _fake_db(row=_audit_row())
    with _repo_with(factory):
        event = PostgresHandoffAuditRepository().create_status_change(
            handoff_id=HANDOFF_ID,
            previous_status=HandoffStatus.PENDING,
            new_status=HandoffStatus.IN_REVIEW,
        )

    assert isinstance(event, HandoffAuditEvent)
    assert event.action is HandoffAuditAction.STATUS_CHANGED
    assert event.previous_status is HandoffStatus.PENDING
# --- Postgres list ----------------------------------------------------------------


def test_postgres_list_parameterized_and_ordered():
    factory, conn, cursor = _fake_db(rows=[_audit_row()])
    with _repo_with(factory):
        events = PostgresHandoffAuditRepository().list_for_handoff(HANDOFF_ID)

    sql, params = cursor.execute.call_args.args
    assert "SELECT" in sql
    assert "FROM handoff_audit_events" in sql
    assert "WHERE handoff_id = %s" in sql
    assert "ORDER BY created_at ASC" in sql
    assert params == (HANDOFF_ID,)
    assert len(events) == 1
    conn.commit.assert_not_called()


def test_postgres_list_no_commit():
    factory, conn, cursor = _fake_db(rows=[])
    with _repo_with(factory):
        PostgresHandoffAuditRepository().list_for_handoff(HANDOFF_ID)
    conn.commit.assert_not_called()


def test_postgres_no_update_or_delete_sql():
    import app.repositories.postgres_handoff_audit_repository as mod

    for sql in (mod._INSERT_AUDIT_SQL, mod._SELECT_AUDIT_SQL):
        upper = sql.upper()
        assert "DELETE" not in upper
        assert "UPDATE" not in upper
        assert "TRUNCATE" not in upper
        assert "DROP" not in upper


def test_postgres_list_maps_events():
    factory, _, _ = _fake_db(rows=[_audit_row()])
    with _repo_with(factory):
        events = PostgresHandoffAuditRepository().list_for_handoff(HANDOFF_ID)
    assert isinstance(events[0], HandoffAuditEvent)
    assert events[0].previous_status is HandoffStatus.PENDING
# --- In-memory audit repository ---------------------------------------------------


def test_memory_append_only_and_filtering():
    repo = InMemoryHandoffAuditRepository()
    other_id = UUID("87654321-4321-8765-4321-876543210000")
    repo.create_status_change(
        handoff_id=HANDOFF_ID,
        previous_status=HandoffStatus.PENDING,
        new_status=HandoffStatus.IN_REVIEW,
    )
    repo.create_status_change(
        handoff_id=other_id,
        previous_status=HandoffStatus.PENDING,
        new_status=HandoffStatus.CANCELLED,
    )

    events = repo.list_for_handoff(HANDOFF_ID)
    assert len(events) == 1
    assert events[0].handoff_id == HANDOFF_ID
    # Other handoff isolated.
    assert len(repo.list_for_handoff(other_id)) == 1


def test_memory_insertion_order():
    repo = InMemoryHandoffAuditRepository()
    for target in (HandoffStatus.IN_REVIEW, HandoffStatus.RESOLVED):
        repo.create_status_change(
            handoff_id=HANDOFF_ID,
            previous_status=HandoffStatus.PENDING,
            new_status=target,
        )
    events = repo.list_for_handoff(HANDOFF_ID)
    assert [e.new_status for e in events] == [
        HandoffStatus.IN_REVIEW,
        HandoffStatus.RESOLVED,
    ]


def test_memory_deep_copy_and_immutable_semantics():
    repo = InMemoryHandoffAuditRepository()
    first = repo.create_status_change(
        handoff_id=HANDOFF_ID,
        previous_status=HandoffStatus.PENDING,
        new_status=HandoffStatus.IN_REVIEW,
    )
    # Mutating a returned (nominally frozen) object must not affect storage.
    object.__setattr__(first, "new_status", HandoffStatus.CANCELLED)
    stored = repo.list_for_handoff(HANDOFF_ID)[0]
    assert stored.new_status is HandoffStatus.IN_REVIEW
    # The returned object itself is a distinct deep copy.
    assert first is not stored