"""Controlled persistence smoke test for the Postgres conversation repository.

Verifies the basic repository roundtrip using a dedicated synthetic customer
key. Never runs automatically — only when explicitly invoked (module CLI or
direct call).

Manual Neon dry-run sequence:

    export DATABASE_URL='...'
    export CONVERSATION_REPOSITORY_BACKEND=postgres

    python -m app.db.migrations.runner
    python -m app.db.smoke_test
"""

from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.repositories.conversation_repository import ConversationRepository
from app.services.phone_normalizer import normalize_customer_phone

SMOKE_TEST_CUSTOMER = "__smoke_test_customer__"

_DELETE_SMOKE_TEST_ROW_SQL = """
DELETE FROM conversation_states
WHERE customer_phone = %s
"""


class PersistenceSmokeTestError(Exception):
    """Raised when persistence smoke test roundtrip fails."""


def _expected_state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        adults=2,
        booking_stage=BookingStage.COLLECTING_DETAILS,
        needs_human=False,
    )


def run_persistence_smoke_test(repository: ConversationRepository) -> None:
    """Verify save/get roundtrip with a synthetic key, then clean up."""
    expected = _expected_state()

    repository.save(SMOKE_TEST_CUSTOMER, expected)
    loaded = repository.get(SMOKE_TEST_CUSTOMER)

    mismatches = [
        field
        for field in ("intent", "tour", "adults", "booking_stage", "needs_human")
        if getattr(loaded, field) != getattr(expected, field)
    ]
    if mismatches:
        raise PersistenceSmokeTestError(
            f"Persistence smoke test roundtrip mismatch in fields: {mismatches}"
        )

    _delete_smoke_test_row()


def _delete_smoke_test_row() -> None:
    """Delete ONLY the synthetic smoke-test row via parameterized SQL."""
    # Imported here to keep module import free of any DB usage.
    from app.db.connection import database_connection

    key = normalize_customer_phone(SMOKE_TEST_CUSTOMER)

    with database_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(_DELETE_SMOKE_TEST_ROW_SQL, (key,))
        conn.commit()


if __name__ == "__main__":
    from app.db.migrations.runner import run_migrations
    from app.repositories.provider import get_conversation_repository

    run_migrations()
    run_persistence_smoke_test(get_conversation_repository())
