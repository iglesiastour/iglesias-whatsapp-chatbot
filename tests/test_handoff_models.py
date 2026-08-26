"""Tests for the human handoff domain model (Phase 6 Step 1)."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import (
    HandoffReason,
    HandoffRequest,
    HandoffStatus,
    create_handoff_request,
    determine_handoff_reason,
)


def _ready_state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )


def test_handoff_reason_enum_values_exact():
    assert {r.value for r in HandoffReason} == {
        "booking_review",
        "human_request",
        "complaint",
        "cancellation_request",
        "existing_booking",
        "safety_escalation",
    }


def test_handoff_status_enum_values_exact():
    assert {s.value for s in HandoffStatus} == {
        "pending",
        "in_review",
        "resolved",
        "cancelled",
    }


def test_ready_for_review_derives_booking_review():
    assert determine_handoff_reason(_ready_state()) is HandoffReason.BOOKING_REVIEW


def test_human_request_intent():
    state = ConversationState(intent=ConversationIntent.HUMAN_REQUEST)
    assert determine_handoff_reason(state) is HandoffReason.HUMAN_REQUEST


def test_complaint_intent():
    state = ConversationState(intent=ConversationIntent.COMPLAINT)
    assert determine_handoff_reason(state) is HandoffReason.COMPLAINT


def test_cancellation_intent():
    state = ConversationState(intent=ConversationIntent.CANCELLATION_REQUEST)
    assert determine_handoff_reason(state) is HandoffReason.CANCELLATION_REQUEST


def test_existing_booking_requiring_human():
    state = ConversationState(
        intent=ConversationIntent.EXISTING_BOOKING, needs_human=True
    )
    assert determine_handoff_reason(state) is HandoffReason.EXISTING_BOOKING


def test_generic_requires_human_safety_escalation():
    state = ConversationState(
        intent=ConversationIntent.GENERAL_QUESTION, needs_human=True
    )
    assert determine_handoff_reason(state) is HandoffReason.SAFETY_ESCALATION


def test_collecting_details_booking_does_not_create_handoff():
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    assert determine_handoff_reason(state) is None


def test_tour_information_does_not_create_handoff():
    state = ConversationState(intent=ConversationIntent.TOUR_INFORMATION, tour="Ephesus")
    assert determine_handoff_reason(state) is None



def test_confirmed_normal_state_is_not_booking_review():
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.CONFIRMED,
    )
    assert determine_handoff_reason(state) is None


def test_cancelled_normal_state_is_not_booking_review():
    state = ConversationState(booking_stage=BookingStage.CANCELLED)
    assert determine_handoff_reason(state) is None


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (ConversationIntent.CANCELLATION_REQUEST, HandoffReason.CANCELLATION_REQUEST),
        (ConversationIntent.COMPLAINT, HandoffReason.COMPLAINT),
    ],
)
def test_intent_outranks_generic_human_flag(intent, expected):
    state = ConversationState(intent=intent, needs_human=True)
    assert determine_handoff_reason(state) is expected


def test_deterministic_repeated_derivation():
    state = _ready_state()
    first = determine_handoff_reason(state)
    for _ in range(5):
        assert determine_handoff_reason(state) is first


def test_default_status_is_pending():
    request = create_handoff_request(
        "+905551112233", _ready_state(), HandoffReason.BOOKING_REVIEW
    )
    assert request.status is HandoffStatus.PENDING


def test_phone_normalized_with_shared_helper():
    request = create_handoff_request(
        "  +90555   111   2233 ", _ready_state(), HandoffReason.BOOKING_REVIEW
    )
    assert request.customer_phone == "+90555 111 2233"


@pytest.mark.parametrize("phone", ["", "   ", "\t\n"])
def test_blank_phone_rejected(phone):
    with pytest.raises(ValidationError):
        create_handoff_request(phone, _ready_state(), HandoffReason.BOOKING_REVIEW)


def test_customer_name_whitespace_normalized_preserving_case():
    request = create_handoff_request(
        "+905551112233",
        _ready_state(),
        HandoffReason.BOOKING_REVIEW,
        customer_name="  Mehmet   Cam  ",
    )
    assert request.customer_name == "Mehmet Cam"


def test_blank_customer_name_becomes_none():
    request = create_handoff_request(
        "+905551112233",
        _ready_state(),
        HandoffReason.BOOKING_REVIEW,
        customer_name="   ",
    )
    assert request.customer_name is None


def test_state_snapshot_independent_from_source_mutation():
    source = _ready_state()
    request = create_handoff_request(
        "+905551112233", source, HandoffReason.BOOKING_REVIEW
    )

    source.tour = "MUTATED"
    source.adults = 99

    assert request.conversation_state.tour == "Ephesus tour"
    assert request.conversation_state.adults == 2


def test_request_is_frozen():
    request = create_handoff_request(
        "+905551112233", _ready_state(), HandoffReason.BOOKING_REVIEW
    )
    assert request.model_config.get("frozen") is True
    with pytest.raises(ValidationError):
        request.customer_phone = "+900000000000"


def test_returned_type_correct():
    request = create_handoff_request(
        "+905551112233", _ready_state(), HandoffReason.BOOKING_REVIEW
    )
    assert isinstance(request, HandoffRequest)
    assert isinstance(request.reason, HandoffReason)


def test_no_raw_prompt_reasoning_or_provider_fields():
    field_names = set(HandoffRequest.model_fields)
    forbidden = {
        "prompt",
        "system_prompt",
        "reasoning",
        "ai_reasoning",
        "safety_patterns",
        "provider_response",
        "raw_response",
        "api_key",
        "database_url",
    }
    assert not (field_names & forbidden)


def test_no_network_or_env_dependency(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "should_not_be_used")
    monkeypatch.setenv("OPENROUTER_API_KEY", "should_not_be_used")
    reason = determine_handoff_reason(_ready_state())
    assert reason is HandoffReason.BOOKING_REVIEW
