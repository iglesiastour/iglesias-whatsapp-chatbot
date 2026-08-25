"""Deterministic tests for the conversation state transition service."""

import os
from datetime import date

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.services.conversation_state_service import apply_message_to_state


def entities_complete() -> dict:
    return {
        "tour": "Ephesus",
        "travel_date": date(2026, 9, 15),
        "adults": 2,
    }


# --- Immutability / object identity ---


def test_incoming_state_not_mutated() -> None:
    state = ConversationState(intent=ConversationIntent.GREETING)
    snapshot = state.model_copy()
    apply_message_to_state(state, "I want to book a tour")
    assert state == snapshot
    assert state.intent is ConversationIntent.GREETING


def test_new_object_returned() -> None:
    state = ConversationState()
    result = apply_message_to_state(state, "hello")
    assert result is not state


def test_result_is_valid_conversation_state() -> None:
    result = apply_message_to_state(ConversationState(), "hello")
    assert isinstance(result, ConversationState)


# --- Intent updates ---


def test_greeting_updates_intent() -> None:
    result = apply_message_to_state(ConversationState(), "hello")
    assert result.intent is ConversationIntent.GREETING


def test_tour_information_updates_intent() -> None:
    result = apply_message_to_state(ConversationState(), "Tell me about Ephesus.")
    assert result.intent is ConversationIntent.TOUR_INFORMATION


def test_classifier_is_actually_used() -> None:
    # Distinct messages must produce their classified intents.
    assert (
        apply_message_to_state(ConversationState(), "how much?").intent
        is ConversationIntent.PRICE_REQUEST
    )
    assert (
        apply_message_to_state(
            ConversationState(), "Cancel my booking"
        ).intent
        is ConversationIntent.CANCELLATION_REQUEST
    )


# --- Booking-stage transitions ---


def test_booking_request_all_fields_missing_is_collecting_details() -> None:
    result = apply_message_to_state(ConversationState(), "I want to book a tour")
    assert result.intent is ConversationIntent.BOOKING_REQUEST
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS


def test_booking_request_partial_details_is_collecting_details() -> None:
    state = ConversationState(tour="Ephesus")
    result = apply_message_to_state(state, "I want to book a tour")
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS


def test_booking_request_complete_details_is_ready_for_review() -> None:
    state = ConversationState(**entities_complete())
    result = apply_message_to_state(state, "I want to book.")
    assert result.intent is ConversationIntent.BOOKING_REQUEST
    assert result.booking_stage is BookingStage.READY_FOR_REVIEW


def test_price_request_during_collecting_preserves_stage() -> None:
    state = ConversationState(booking_stage=BookingStage.COLLECTING_DETAILS)
    result = apply_message_to_state(state, "How much does it cost?")
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS
    assert result.intent is ConversationIntent.PRICE_REQUEST


def test_availability_request_during_collecting_preserves_stage() -> None:
    state = ConversationState(booking_stage=BookingStage.COLLECTING_DETAILS)
    result = apply_message_to_state(state, "Is it available on Monday?")
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS
    assert result.intent is ConversationIntent.AVAILABILITY_REQUEST


@pytest.mark.parametrize("intent_message", ["This is unacceptable.", "I want to cancel"])
@pytest.mark.parametrize("stage", [BookingStage.COLLECTING_DETAILS, BookingStage.READY_FOR_REVIEW])
def test_human_required_intent_during_active_booking_is_human_review(
    stage: BookingStage, intent_message: str
) -> None:
    state = ConversationState(booking_stage=stage)
    result = apply_message_to_state(state, intent_message)
    assert result.booking_stage is BookingStage.HUMAN_REVIEW


# --- Human escalation ---


@pytest.mark.parametrize(
    "message",
    ["This is unacceptable.", "I want to talk to a human.", "Cancel my booking"],
)
def test_human_required_intents_set_needs_human(message: str) -> None:
    result = apply_message_to_state(ConversationState(), message)
    assert result.needs_human is True


def test_existing_needs_human_true_never_reset() -> None:
    state = ConversationState(needs_human=True)
    result = apply_message_to_state(state, "Tell me about Ephesus.")
    assert result.needs_human is True


def test_normal_intent_with_needs_human_false_stays_false() -> None:
    result = apply_message_to_state(
        ConversationState(), "Tell me about Ephesus."
    )
    assert result.needs_human is False


# --- CONFIRMED / CANCELLED protection ---


@pytest.mark.parametrize("stage", [BookingStage.CONFIRMED, BookingStage.CANCELLED])
@pytest.mark.parametrize(
    "message",
    [
        "I want to book a tour",
        "This is unacceptable.",
        "I want to talk to a human.",
        "Cancel my booking",
        "How much?",
    ],
)
def test_authoritative_stages_never_changed(stage: BookingStage, message: str) -> None:
    state = ConversationState(**entities_complete(), booking_stage=stage)
    result = apply_message_to_state(state, message)
    assert result.booking_stage is stage


# --- Existing booking & ordinary intents ---


def test_existing_booking_intent_preserves_booking_stage() -> None:
    for stage in (
        BookingStage.NONE,
        BookingStage.COLLECTING_DETAILS,
        BookingStage.READY_FOR_REVIEW,
    ):
        state = ConversationState(booking_stage=stage)
        result = apply_message_to_state(state, "What time is my reservation?")
        assert result.intent is ConversationIntent.EXISTING_BOOKING
        assert result.booking_stage is stage


def test_normal_question_with_none_stage_preserves_none() -> None:
    result = apply_message_to_state(
        ConversationState(), "What are your office hours?"
    )
    assert result.booking_stage is BookingStage.NONE
    assert result.intent is ConversationIntent.GENERAL_QUESTION


def test_price_request_with_none_stage_preserves_none() -> None:
    result = apply_message_to_state(ConversationState(), "How much is the tour?")
    assert result.booking_stage is BookingStage.NONE
    assert result.intent is ConversationIntent.PRICE_REQUEST


def test_availability_request_with_none_stage_preserves_none() -> None:
    result = apply_message_to_state(
        ConversationState(), "Is it available on Monday?"
    )
    assert result.booking_stage is BookingStage.NONE
    assert result.intent is ConversationIntent.AVAILABILITY_REQUEST


# --- Entity preservation ---


def test_all_entity_fields_remain_unchanged() -> None:
    state = ConversationState(
        tour="Ephesus",
        travel_date=date(2026, 9, 15),
        adults=2,
        children=1,
        cruise_ship="Celebrity Equinox",
        hotel="Korumar Hotel",
        pickup_location="Hotel lobby",
        preferred_language="English",
    )
    snapshot = state.model_copy()
    result = apply_message_to_state(state, "I want to book a tour")

    assert result.tour == snapshot.tour
    assert result.travel_date == snapshot.travel_date
    assert result.adults == snapshot.adults
    assert result.children == snapshot.children
    assert result.cruise_ship == snapshot.cruise_ship
    assert result.hotel == snapshot.hotel
    assert result.pickup_location == snapshot.pickup_location
    assert result.preferred_language == snapshot.preferred_language

    # The original object was not touched either.
    assert state == snapshot


# --- Determinism ---


def test_repeated_calls_yield_equivalent_results() -> None:
    state = ConversationState(tour="Ephesus")
    first = apply_message_to_state(state, "I want to book a tour")
    second = apply_message_to_state(state, "I want to book a tour")
    assert first == second


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    apply_message_to_state(ConversationState(), "hello")
    assert dict(os.environ) == snapshot

