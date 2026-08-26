"""Tests for handoff request/row mapping (no database, no network)."""

from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import HandoffReason, HandoffStatus, HandoffRequest
from app.repositories.handoff_mapping import (
    db_row_to_persisted_handoff,
    handoff_request_to_db_values,
)

KEY = "a" * 64


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


def test_request_to_db_values_serializes_enums():
    values = handoff_request_to_db_values(uuid4(), KEY, _request())
    assert values["reason"] == "booking_review"
    assert values["status"] == "pending"
    assert values["intent"] == "booking_request"
    assert values["booking_stage"] == "ready_for_review"
    assert values["idempotency_key"] == KEY


def test_request_to_db_values_preserves_date_and_nulls():
    values = handoff_request_to_db_values(uuid4(), KEY, _request())
    assert values["travel_date"] == date(2026, 9, 10)
    assert values["adults"] == 2
    assert values["children"] == 1
    assert values["needs_human"] is True


def test_request_to_db_values_handles_missing_optional_fields():
    request = HandoffRequest(
        customer_phone="+905551112233",
        reason=HandoffReason.SAFETY_ESCALATION,
        conversation_state=ConversationState(),
    )
    values = handoff_request_to_db_values(uuid4(), KEY, request)
    assert values["customer_name"] is None
    assert values["tour"] is None
    assert values["travel_date"] is None
    assert values["adults"] is None
    assert values["cruise_ship"] is None
    assert values["status"] == "pending"


def test_uuid_is_in_values():
    handoff_id = uuid4()
    values = handoff_request_to_db_values(handoff_id, KEY, _request())
    assert values["id"] == handoff_id


def test_db_row_roundtrip():
    row = handoff_request_to_db_values(uuid4(), KEY, _request())
    persisted = db_row_to_persisted_handoff(row)

    assert persisted.idempotency_key == KEY
    assert persisted.customer_phone == "+90555 111 2233"
    assert persisted.reason is HandoffReason.BOOKING_REVIEW
    assert persisted.status is HandoffStatus.PENDING
    assert persisted.conversation_state.intent is ConversationIntent.BOOKING_REQUEST
    assert persisted.conversation_state.tour == "Ephesus tour"
    assert persisted.conversation_state.travel_date == date(2026, 9, 10)
    assert persisted.conversation_state.booking_stage is BookingStage.READY_FOR_REVIEW
    assert persisted.conversation_state.needs_human is True


def test_db_row_roundtrip_preserves_null_optionals():
    row = handoff_request_to_db_values(
        uuid4(),
        KEY,
        HandoffRequest(
            customer_phone="+905551112233",
            customer_name="  ",
            reason=HandoffReason.SAFETY_ESCALATION,
            conversation_state=ConversationState(),
        ),
    )
    persisted = db_row_to_persisted_handoff(row)
    assert persisted.customer_name is None
    assert persisted.conversation_state.tour is None


def _reject_row(**overrides):
    row = {
        "id": uuid4(),
        "idempotency_key": KEY,
        "customer_phone": "+905551112233",
        "customer_name": None,
        "reason": "booking_review",
        "status": "pending",
        "intent": "tour_information",
        "tour": None,
        "travel_date": None,
        "adults": None,
        "children": None,
        "cruise_ship": None,
        "hotel": None,
        "pickup_location": None,
        "preferred_language": None,
        "booking_stage": "none",
        "needs_human": False,
    }
    row.update(overrides)
    return row


def test_invalid_intent_value_rejected_not_repaired():
    with pytest.raises(ValueError):
        db_row_to_persisted_handoff(_reject_row(intent="not_a_real_intent"))


def test_invalid_reason_value_rejected():
    with pytest.raises(ValueError):
        db_row_to_persisted_handoff(_reject_row(reason="bogus"))


def test_invalid_adults_rejected_by_state_validation():
    with pytest.raises(ValidationError):
        db_row_to_persisted_handoff(_reject_row(adults=500))


def test_timestamp_keys_ignored():
    row = _reject_row(created_at="ignored", updated_at="ignored")
    persisted = db_row_to_persisted_handoff(row)
    assert not hasattr(persisted, "created_at")


def test_mapping_is_deterministic():
    a = handoff_request_to_db_values(uuid4(), KEY, _request())
    b = handoff_request_to_db_values(uuid4(), KEY, _request())
    a.pop("id")
    b.pop("id")
    assert a == b


def test_no_network_or_env_dependency(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "should_not_be_used")
    monkeypatch.setenv("DATABASE_URL", "should_not_be_used")
    values = handoff_request_to_db_values(uuid4(), KEY, _request())
    assert values["reason"] == "booking_review"
