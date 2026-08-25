"""Tests for the persistence smoke-test utility (no real database)."""

from unittest.mock import MagicMock, patch

import pytest

from app.db.connection import DatabaseNotConfiguredError
from app.db.smoke_test import (
    SMOKE_TEST_CUSTOMER,
    PersistenceSmokeTestError,
    run_persistence_smoke_test,
)
from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.repositories.conversation_repository import ConversationRepository


class RecordingRepository(ConversationRepository):
    """In-memory double recording save/get calls."""

    def __init__(
        self,
        state: ConversationState | None = None,
        fixed_get_state: ConversationState | None = None,
    ):
        self.saved: list[tuple[str, ConversationState]] = []
        self.get_calls: list[str] = []
        self.state = state if state is not None else ConversationState()
        self.fixed_get_state = fixed_get_state

    def get(self, customer_phone: str) -> ConversationState:
        self.get_calls.append(customer_phone)
        if self.fixed_get_state is not None:
            return self.fixed_get_state.model_copy()
        return self.state.model_copy()

    def save(self, customer_phone: str, state: ConversationState) -> None:
        self.saved.append((customer_phone, state.model_copy()))
        if self.fixed_get_state is None:
            self.state = state.model_copy()

    def clear(self) -> None:
        pass


def test_smoke_test_saves_with_synthetic_key(fake_db) -> None:
    repository = RecordingRepository()
    run_persistence_smoke_test(repository)

    assert repository.saved[0][0] == SMOKE_TEST_CUSTOMER
    assert SMOKE_TEST_CUSTOMER.startswith("__smoke_test_")


def test_saved_state_contains_expected_fields(fake_db) -> None:
    repository = RecordingRepository()
    run_persistence_smoke_test(repository)

    saved = repository.saved[0][1]
    assert saved.intent is ConversationIntent.BOOKING_REQUEST
    assert saved.tour == "Ephesus"
    assert saved.adults == 2
    assert saved.booking_stage is BookingStage.COLLECTING_DETAILS
    assert saved.needs_human is False


def test_smoke_test_gets_same_synthetic_key(fake_db) -> None:
    repository = RecordingRepository()
    run_persistence_smoke_test(repository)

    assert repository.get_calls == [SMOKE_TEST_CUSTOMER]


def test_correct_roundtrip_passes(fake_db) -> None:
    provider_state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        adults=2,
        booking_stage=BookingStage.COLLECTING_DETAILS,
        needs_human=False,
    )
    repository = RecordingRepository(state=provider_state)
    # Should not raise.
    run_persistence_smoke_test(repository)


def test_mismatched_roundtrip_raises_persistence_error(fake_db) -> None:
    wrong = ConversationState(intent=ConversationIntent.GREETING)
    repository = RecordingRepository(fixed_get_state=wrong)
    with pytest.raises(PersistenceSmokeTestError, match="mismatch"):
        run_persistence_smoke_test(repository)


# --- Cleanup ---


def test_cleanup_targets_only_smoke_test_key() -> None:
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("app.db.connection.database_connection") as factory:
        factory.return_value.__enter__ = MagicMock(return_value=conn)
        factory.return_value.__exit__ = MagicMock(return_value=False)
        repository = RecordingRepository()
        run_persistence_smoke_test(repository)

    sql, params = cursor.execute.call_args.args
    assert params == (SMOKE_TEST_CUSTOMER,)


def test_cleanup_sql_is_parameterized_delete() -> None:
    from app.db import smoke_test

    assert "%s" in smoke_test._DELETE_SMOKE_TEST_ROW_SQL
    upper = smoke_test._DELETE_SMOKE_TEST_ROW_SQL.upper()
    assert "DELETE FROM CONVERSATION_STATES" in upper
    assert "TRUNCATE" not in upper and "DROP" not in upper


def test_cleanup_commits_once() -> None:
    repository = RecordingRepository()

    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("app.db.connection.database_connection") as factory:
        factory.return_value.__enter__ = MagicMock(return_value=conn)
        factory.return_value.__exit__ = MagicMock(return_value=False)
        run_persistence_smoke_test(repository)

    assert conn.commit.call_count == 1


# --- Errors / isolation ---


def test_database_errors_propagate_unchanged() -> None:
    repository = RecordingRepository()

    with patch(
        "app.db.connection.get_database_connection",
        side_effect=DatabaseNotConfiguredError("DATABASE_URL is not configured."),
    ):
        with pytest.raises(DatabaseNotConfiguredError):
            run_persistence_smoke_test(repository)


@pytest.fixture()
def fake_db():
    """Patch database_connection for smoke-test cleanup; returns (conn, cursor)."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with patch("app.db.connection.database_connection") as factory:
        factory.return_value.__enter__ = MagicMock(return_value=conn)
        factory.return_value.__exit__ = MagicMock(return_value=False)
        yield conn, cursor


def test_smoke_test_not_run_on_import() -> None:
    # Importing the module must have zero side effects; verify no connection
    # attempt is made simply by importing it fresh from sys.modules.
    import importlib
    import sys
    from unittest.mock import MagicMock as _MagicMock

    fake_connect = _MagicMock()
    with patch("app.db.connection.psycopg.connect", fake_connect):
        module = sys.modules.get("app.db.smoke_test")
        if module is None:
            module = importlib.import_module("app.db.smoke_test")
        else:
            importlib.reload(module)

    assert fake_connect.call_count == 0


def test_no_network_or_environment_dependency(fake_db) -> None:
    import os

    snapshot = dict(os.environ)
    repository = RecordingRepository(
        state=ConversationState(
            intent=ConversationIntent.BOOKING_REQUEST,
            tour="Ephesus",
            adults=2,
            booking_stage=BookingStage.COLLECTING_DETAILS,
        )
    )
    run_persistence_smoke_test(repository)
    assert dict(os.environ) == snapshot
