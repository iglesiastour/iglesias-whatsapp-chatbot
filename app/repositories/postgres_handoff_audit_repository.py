"""PostgreSQL-backed append-only handoff audit repository.

Uses the shared database_connection() helper. Append-only: only INSERT and
parameterized SELECT are issued; no UPDATE/DELETE ever.
"""

from uuid import UUID, uuid4

from psycopg.rows import dict_row

from app.db.connection import database_connection
from app.models.handoff import HandoffStatus
from app.models.handoff_audit import (
    HandoffAuditAction,
    HandoffAuditEvent,
)
from app.repositories.handoff_audit_repository import HandoffAuditRepository

_AUDIT_COLUMNS = (
    "id, handoff_id, action, previous_status, new_status, created_at"
)

_INSERT_AUDIT_SQL = (
    "INSERT INTO handoff_audit_events ("
    "id, handoff_id, action, previous_status, new_status"
    ") VALUES ("
    "%(id)s, %(handoff_id)s, %(action)s, %(previous_status)s, %(new_status)s"
    ") RETURNING " + _AUDIT_COLUMNS
)

_SELECT_AUDIT_SQL = (
    "SELECT " + _AUDIT_COLUMNS + " FROM handoff_audit_events "
    "WHERE handoff_id = %s ORDER BY created_at ASC, id ASC"
)


def _row_to_event(row: dict) -> HandoffAuditEvent:
    from datetime import datetime

    created = row["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    return HandoffAuditEvent(
        id=UUID(str(row["id"])),
        handoff_id=UUID(str(row["handoff_id"])),
        action=HandoffAuditAction(row["action"]),
        previous_status=HandoffStatus(row["previous_status"]),
        new_status=HandoffStatus(row["new_status"]),
        created_at=created,
    )


class PostgresHandoffAuditRepository(HandoffAuditRepository):
    def create_status_change(
        self,
        *,
        handoff_id: UUID,
        previous_status: HandoffStatus,
        new_status: HandoffStatus,
    ) -> HandoffAuditEvent:
        values = {
            "id": uuid4(),
            "handoff_id": handoff_id,
            "action": HandoffAuditAction.STATUS_CHANGED.value,
            "previous_status": previous_status.value,
            "new_status": new_status.value,
        }

        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_INSERT_AUDIT_SQL, values)
                row = cursor.fetchone()
            conn.commit()

        assert row is not None
        return _row_to_event(row)

    def list_for_handoff(
        self,
        handoff_id: UUID,
    ) -> list[HandoffAuditEvent]:
        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_SELECT_AUDIT_SQL, (handoff_id,))
                rows = cursor.fetchall()

        return [_row_to_event(row) for row in rows]