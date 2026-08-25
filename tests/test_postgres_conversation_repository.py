"""Tests for PostgresConversationRepository using fake DB connections."""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.postgres_conversation_repository import (
    PostgresConversationRepository,
)


def _fake_db(row=None):
    """Return (factory, conn, cursor) mocks wired as context managers."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = row
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor

    factory = MagicMock()
    factory.return_value.__enter__ = MagicMock(return_value=conn)
    factory.return_value.__exit__ = MagicMock(return_value=False)
    return factory, conn, cursor


def repo_with(factory):
    return patch(
        "app.repositories.postgres_conversation_repository.database_connection",
        factory,
    )


FULL_ROW = {
    "customer_phone": "+905551112233",
    "intent": "booking_request",
    "tour": "Ephesus",
    "travel_date": date(2026, 9, 10),
    "adults": 2,
    "children": 1,
    "cruise_ship": "Equinox",
    "hotel": "Korumar",
    "pickup_location": "Port",
    "preferred_language": "English",
    "booking_stage": "ready_for_review",
    "needs_human": False,
}


# --- Contract / GET ---


def test_repository_subclasses_conversation_repository() -> None:
    assert issubclass(PostgresConversationRepository, ConversationRepository)
    assert isinstance(PostgresConversationRepository(), ConversationRepository)


def test_get_normalizes_phone() -> None:
    factory, _, cursor = _fake_db(row=FULL_ROW)
    with repo_with(factory):
        PostgresConversationRepository().get("  +90555 111 2233  ")

    assert cursor.execute.call_args.args[1] == ("+90555 111 2233",)


def test_select_is_parameterized_and_phone_not_interpolated() -> None:
    phone = "'; DROP TABLE conversation_states; --"
    factory, _, cursor = _fake_db(row=None)
    with repo_with(factory):
        PostgresConversationRepository().get(phone)

    sql = cursor.execute.call_args.args[0]
    assert "%s" in sql
    assert phone not in sql


def test_correct_parameter_passed_to_execute() -> None:
    factory, _, cursor = _fake_db(row=None)
    with repo_with(factory):
        PostgresConversationRepository().get("+905551112233")

    assert cursor.execute.call_args.args[1] == ("+905551112233",)


def test_existing_row_maps_to_conversation_state() -> None:
    factory, _, _ = _fake_db(row=dict(FULL_ROW))
    with repo_with(factory):
        state = PostgresConversationRepository().get("+905551112233")

    assert state.intent is ConversationIntent.BOOKING_REQUEST
    assert state.tour == "Ephesus"
    assert state.travel_date == date(2026, 9, 10)


def test_missing_row_returns_default_state() -> None:
    factory, _, _ = _fake_db(row=None)
    with repo_with(factory):
        state = PostgresConversationRepository().get("+900000000000")

    assert state == ConversationState()


def test_get_does_not_commit_or_insert() -> None:
    factory, conn, cursor = _fake_db(row=None)
    with repo_with(factory):
        PostgresConversationRepository().get("+905551112233")

    conn.commit.assert_not_called()
    executed_sql = cursor.execute.call_args.args[0]
    assert "INSERT" not in executed_sql.upper()


# --- SAVE ---


def test_save_normalizes_phone() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save(
            "  +90555 111 2233  ", ConversationState()
        )

    params = cursor.execute.call_args.args[1]
    assert params["customer_phone"] == "+90555 111 2233"


def test_upsert_sql_contains_on_conflict() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", ConversationState())

    sql = cursor.execute.call_args.args[0]
    assert "ON CONFLICT (customer_phone)" in sql


def test_save_uses_mapping_parameters() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", ConversationState())

    args = cursor.execute.call_args.args
    assert len(args) == 2
    assert isinstance(args[1], dict)


def test_save_passes_all_state_fields() -> None:
    factory, _, cursor = _fake_db()
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
    )
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", state)

    params = cursor.execute.call_args.args[1]
    for key in (
        "customer_phone", "intent", "tour", "travel_date", "adults", "children",
        "cruise_ship", "hotel", "pickup_location", "preferred_language",
        "booking_stage", "needs_human",
    ):
        assert key in params


def test_save_serializes_enum_values() -> None:
    factory, _, cursor = _fake_db()
    state = ConversationState(
        intent=ConversationIntent.PRICE_REQUEST,
        booking_stage=BookingStage.HUMAN_REVIEW,
    )
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", state)

    params = cursor.execute.call_args.args[1]
    assert params["intent"] == "price_request"
    assert params["booking_stage"] == "human_review"


def test_updated_at_now_present_in_upsert() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", ConversationState())

    sql = cursor.execute.call_args.args[0]
    assert "updated_at = NOW()" in sql


def test_created_at_not_overwritten_in_update_clause() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", ConversationState())

    sql = cursor.execute.call_args.args[0]
    update_clause = sql.split("DO UPDATE SET")[1]
    assert "created_at" not in update_clause


def test_commit_called_exactly_once_on_success() -> None:
    factory, conn, _ = _fake_db()
    with repo_with(factory):
        PostgresConversationRepository().save("+905551112233", ConversationState())

    assert conn.commit.call_count == 1


def test_execute_failure_propagates_and_no_commit() -> None:
    factory, conn, cursor = _fake_db()
    cursor.execute.side_effect = RuntimeError("db exploded")

    with repo_with(factory):
        with pytest.raises(RuntimeError, match="db exploded"):
            PostgresConversationRepository().save("+905551112233", ConversationState())

    conn.commit.assert_not_called()


# --- CLEAR ---


def test_clear_raises_not_implemented() -> None:
    repo = PostgresConversationRepository()
    with pytest.raises(NotImplementedError, match="not supported"):
        repo.clear()


def test_clear_executes_no_sql() -> None:
    factory, _, cursor = _fake_db()
    with repo_with(factory):
        try:
            PostgresConversationRepository().clear()
        except NotImplementedError:
            pass

    cursor.execute.assert_not_called()


def test_clear_never_deletes_or_truncates() -> None:
    import app.repositories.postgres_conversation_repository as module

    source_sql = (module._UPSERT_SQL + module._SELECT_SQL).upper()
    for forbidden in ("DELETE FROM", "TRUNCATE", "DROP"):
        assert forbidden not in source_sql


# --- Architecture / safety ---


def test_uses_database_connection_abstraction() -> None:
    import inspect

    import app.repositories.postgres_conversation_repository as module

    source = inspect.getsource(module)
    assert "database_connection()" in source
    assert "psycopg.connect" not in source


def test_no_openrouter_safeai_route_dependencies() -> None:
    import sys

    module = sys.modules[PostgresConversationRepository.__module__]
    assert not hasattr(module, "OpenRouterProvider")
    assert not hasattr(module, "SafeAIService")
    assert not hasattr(module, "router")


def test_no_environment_dependency() -> None:
    import os

    snapshot = dict(os.environ)
    factory, _, _ = _fake_db()
    with repo_with(factory):
        try:
            PostgresConversationRepository().clear()
        except NotImplementedError:
            pass
    assert dict(os.environ) == snapshot

