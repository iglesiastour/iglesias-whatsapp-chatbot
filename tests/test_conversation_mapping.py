"""Deterministic tests for ConversationState ↔ DB row mapping."""

import os
from datetime import date

import pytest
from pydantic import ValidationError

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.repositories.conversation_mapping import (
    db_row_to_state,
    state_to_db_values,
)

EXPECTED_KEYS = {
    "customer_phone",
    "intent",
    "tour",
    "travel_date",
    "adults",
    "children",
    "cruise_ship",
    "hotel",
    "pickup_location",
    "preferred_language",
    "booking_stage",
    "needs_human",
}


def full_state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
        booking_stage=BookingStage.COLLECTING_DETAILS,
        needs_human=True,
    )


def test_state_to_db_values_includes_exactly_expected_keys() -> None:
    values = state_to_db_values("+905551112233", full_state())
    assert set(values.keys()) == EXPECTED_KEYS


def test_intent_serialized_via_value() -> None:
    values = state_to_db_values("p", full_state())
    assert values["intent"] == "booking_request"


def test_booking_stage_serialized_via_value() -> None:
    values = state_to_db_values("p", full_state())
    assert values["booking_stage"] == "collecting_details"


def test_date_preserved() -> None:
    values = state_to_db_values("p", full_state())
    assert values["travel_date"] == date(2026, 9, 10)
    assert isinstance(values["travel_date"], date)


def test_nullable_fields_preserved() -> None:
    values = state_to_db_values("p", ConversationState())
    for field in ("tour", "travel_date", "adults", "children", "cruise_ship", "hotel", "pickup_location", "preferred_language"):
        assert values[field] is None


def test_needs_human_preserved() -> None:
    assert state_to_db_values("p", ConversationState(needs_human=True))["needs_human"] is True
    assert state_to_db_values("p", ConversationState())["needs_human"] is False


def test_row_to_state_valid() -> None:
    row = {
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
        # extra timestamp keys must be ignored
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
    }
    state = db_row_to_state(row)
    assert isinstance(state, ConversationState)
    assert state.intent is ConversationIntent.BOOKING_REQUEST
    assert state.booking_stage is BookingStage.READY_FOR_REVIEW
    assert state.travel_date == date(2026, 9, 10)
    assert state.adults == 2


@pytest.mark.parametrize("intent", list(ConversationIntent))
def test_all_intent_values_roundtrip(intent: ConversationIntent) -> None:
    state = ConversationState(intent=intent)
    row = state_to_db_values("p", state)
    assert db_row_to_state(row).intent is intent


@pytest.mark.parametrize("stage", list(BookingStage))
def test_all_booking_stage_values_roundtrip(stage: BookingStage) -> None:
    state = ConversationState(booking_stage=stage)
    row = state_to_db_values("p", state)
    assert db_row_to_state(row).booking_stage is stage


def test_invalid_intent_raises() -> None:
    row = state_to_db_values("p", ConversationState())
    row["intent"] = "not-a-real-intent"
    with pytest.raises(ValueError):
        db_row_to_state(row)


def test_invalid_booking_stage_raises() -> None:
    row = state_to_db_values("p", ConversationState())
    row["booking_stage"] = "not-a-real-stage"
    with pytest.raises(ValueError):
        db_row_to_state(row)


def test_no_network_or_environment_dependency() -> None:
    snapshot = dict(os.environ)
    state_to_db_values("+905551112233", full_state())
    db_row_to_state(state_to_db_values("p", full_state()))
    assert dict(os.environ) == snapshot
