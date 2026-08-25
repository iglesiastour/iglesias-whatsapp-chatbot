"""Deterministic tests for conversation intent/state domain models."""

import pytest
from pydantic import ValidationError

from app.models.conversation import BookingStage, ConversationIntent, ConversationState


def test_default_state() -> None:
    state = ConversationState()
    assert state.intent is ConversationIntent.GENERAL_QUESTION
    assert state.booking_stage is BookingStage.NONE
    assert state.needs_human is False
    assert state.tour is None
    assert state.travel_date is None
    assert state.adults is None
    assert state.children is None


def test_every_intent_enum_value() -> None:
    assert {intent.value for intent in ConversationIntent} == {
        "greeting",
        "general_question",
        "tour_information",
        "price_request",
        "availability_request",
        "booking_request",
        "existing_booking",
        "cancellation_request",
        "complaint",
        "human_request",
        "unsupported",
    }


def test_every_booking_stage_enum_value() -> None:
    assert {stage.value for stage in BookingStage} == {
        "none",
        "collecting_details",
        "ready_for_review",
        "human_review",
        "confirmed",
        "cancelled",
    }


@pytest.mark.parametrize("adults", [1, 50, 100])
def test_valid_adults_bounds(adults: int) -> None:
    assert ConversationState(adults=adults).adults == adults


@pytest.mark.parametrize("adults", [0, -1, 101, 1000])
def test_invalid_adults_rejected(adults: int) -> None:
    with pytest.raises(ValidationError):
        ConversationState(adults=adults)


@pytest.mark.parametrize("children", [0, 5, 100])
def test_valid_children_bounds(children: int) -> None:
    assert ConversationState(children=children).children == children


@pytest.mark.parametrize("children", [-1, 101])
def test_invalid_children_rejected(children: int) -> None:
    with pytest.raises(ValidationError):
        ConversationState(children=children)


def test_adults_zero_is_not_auto_converted() -> None:
    with pytest.raises(ValidationError):
        ConversationState(intent=ConversationIntent.BOOKING_REQUEST, adults=0)


def test_missing_booking_fields_all_missing() -> None:
    state = ConversationState(intent=ConversationIntent.BOOKING_REQUEST)
    assert state.missing_booking_fields() == ("tour", "travel_date", "adults")


def test_missing_booking_fields_partially_complete() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
    )
    assert state.missing_booking_fields() == ("travel_date", "adults")


def test_missing_booking_fields_complete_returns_empty() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date="2026-09-15",
        adults=2,
    )
    assert state.missing_booking_fields() == ()


def test_optional_booking_fields_not_required() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date="2026-09-15",
        adults=2,
        children=None,
        hotel=None,
        cruise_ship=None,
        pickup_location=None,
        preferred_language=None,
    )
    assert state.is_booking_ready


@pytest.mark.parametrize(
    "intent",
    [
        ConversationIntent.GREETING,
        ConversationIntent.TOUR_INFORMATION,
        ConversationIntent.PRICE_REQUEST,
    ],
)
def test_non_booking_intent_is_never_booking_ready(intent: ConversationIntent) -> None:
    state = ConversationState(
        intent=intent,
        tour="Ephesus",
        travel_date="2026-09-15",
        adults=2,
    )
    assert not state.is_booking_ready


def test_booking_ready_true_only_with_required_fields() -> None:
    incomplete = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST, tour="Ephesus"
    )
    complete = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date="2026-09-15",
        adults=2,
    )
    assert not incomplete.is_booking_ready
    assert complete.is_booking_ready


def test_explicit_needs_human_true_requires_human() -> None:
    state = ConversationState(needs_human=True)
    assert state.requires_human


@pytest.mark.parametrize(
    "intent",
    [
        ConversationIntent.COMPLAINT,
        ConversationIntent.HUMAN_REQUEST,
        ConversationIntent.CANCELLATION_REQUEST,
    ],
)
def test_special_intents_require_human(intent: ConversationIntent) -> None:
    assert ConversationState(intent=intent).requires_human


def test_tour_information_does_not_require_human() -> None:
    assert not ConversationState(
        intent=ConversationIntent.TOUR_INFORMATION
    ).requires_human


def test_booking_request_does_not_automatically_require_human() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    assert not state.requires_human


def test_travel_date_parses_iso_input() -> None:
    from datetime import date

    state = ConversationState(travel_date="2026-09-15")
    assert state.travel_date == date(2026, 9, 15)


def test_no_environment_dependency() -> None:
    import os

    snapshot = dict(os.environ)
    ConversationState(tour="Ephesus")
    assert dict(os.environ) == snapshot
