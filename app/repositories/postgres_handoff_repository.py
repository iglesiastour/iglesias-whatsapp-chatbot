"""PostgreSQL-backed HandoffRepository implementation.

Uses explicit parameterized SQL through the shared database_connection()
helper. UUIDs are generated in Python (no DB extension required). A UNIQUE
constraint on idempotency_key prevents duplicate review tasks at the DB
layer; concurrent INSERT races are recovered by re-fetching the existing row.
"""

from uuid import UUID, uuid4

from psycopg import errors as psycopg_errors
from psycopg.rows import dict_row

from app.db.connection import database_connection
from app.models.handoff import (
    HandoffReason,
    HandoffRequest,
    HandoffStatus,
    PersistedHandoff,
)
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
    HandoffRepositoryDuplicateError,
)
from app.repositories.handoff_mapping import (
    db_row_to_persisted_handoff,
    handoff_request_to_db_values,
)

_COLUMN_LIST = (
    "id, idempotency_key, customer_phone, customer_name, reason, status, intent, "
    "tour, travel_date, adults, children, cruise_ship, hotel, "
    "pickup_location, preferred_language, booking_stage, needs_human"
)

_VALUES_LIST = (
    "%(id)s, %(idempotency_key)s, %(customer_phone)s, %(customer_name)s, "
    "%(reason)s, %(status)s, %(intent)s, %(tour)s, %(travel_date)s, %(adults)s, "
    "%(children)s, %(cruise_ship)s, %(hotel)s, %(pickup_location)s, "
    "%(preferred_language)s, %(booking_stage)s, %(needs_human)s"
)

_INSERT_SQL = "INSERT INTO handoff_requests (" + _COLUMN_LIST + ") VALUES (" + _VALUES_LIST + ") RETURNING " + _COLUMN_LIST

_SELECT_SQL = "SELECT " + _COLUMN_LIST + " FROM handoff_requests WHERE id = %s"

_SELECT_BY_KEY_SQL = "SELECT " + _COLUMN_LIST + " FROM handoff_requests WHERE idempotency_key = %s"

_UPDATE_STATUS_SQL = (
    "UPDATE handoff_requests "
    "SET status = %s, updated_at = NOW() "
    "WHERE id = %s "
    "RETURNING " + _COLUMN_LIST
)

_LIST_SQL_BASE = "SELECT " + _COLUMN_LIST + " FROM handoff_requests"


class PostgresHandoffRepository(HandoffRepository):
    def create(
        self,
        request: HandoffRequest,
        idempotency_key: str,
    ) -> PersistedHandoff:
        values = handoff_request_to_db_values(uuid4(), idempotency_key, request)

        try:
            with database_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(_INSERT_SQL, values)
                    row = cursor.fetchone()
                conn.commit()
        except psycopg_errors.UniqueViolation as exc:
            existing = self.get_by_idempotency_key(idempotency_key)
            if existing is not None:
                return existing
            raise HandoffRepositoryDuplicateError(
                "Handoff already exists for idempotency_key: " + idempotency_key
            ) from exc

        assert row is not None
        return db_row_to_persisted_handoff(row)

    def get(self, handoff_id: UUID) -> PersistedHandoff | None:
        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_SELECT_SQL, (handoff_id,))
                row = cursor.fetchone()

        if row is None:
            return None
        return db_row_to_persisted_handoff(row)

    def get_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> PersistedHandoff | None:
        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_SELECT_BY_KEY_SQL, (idempotency_key,))
                row = cursor.fetchone()

        if row is None:
            return None
        return db_row_to_persisted_handoff(row)

    def update_status(
        self,
        handoff_id: UUID,
        status: HandoffStatus,
    ) -> PersistedHandoff:
        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_UPDATE_STATUS_SQL, (status.value, handoff_id))
                row = cursor.fetchone()
            if row is None:
                # Missing row: nothing to commit.
                raise HandoffNotFoundError()
            conn.commit()

        return db_row_to_persisted_handoff(row)

    def list_handoffs(
        self,
        *,
        status: HandoffStatus | None = None,
        reason: HandoffReason | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PersistedHandoff]:
        """List handoffs with optional status/reason filters, fixed order.

        Only parameterized values are dynamic; the SQL text is assembled from
        static clause strings (never interpolated query values).
        """
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = %s")
            params.append(status.value)
        if reason is not None:
            clauses.append("reason = %s")
            params.append(reason.value)

        sql = _LIST_SQL_BASE
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at ASC, id ASC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()

        return [db_row_to_persisted_handoff(row) for row in rows]

