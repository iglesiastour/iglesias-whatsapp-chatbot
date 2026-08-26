"""Tests for the HandoffReview read model (Phase 6 Step 6)."""

from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import (
    HandoffReason,
    HandoffStatus,
    PersistedHandoff,
)
from app.models.handoff_review import (
    HandoffReview,
    build_handoff_review,
    build_handoff_review_summary,
)


def _full_review() -> HandoffReview:
    return HandoffReview(
        handoff_id=UUID("12345678123456781234567812345678"),
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        status=HandoffStatus.PENDING,
        intent=ConversationIntent.BOOKING_REQUEST,
        booking_stage=BookingStage.READY_FOR_REVIEW,
        needs_human=True,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
    )


def _persisted(
    status: HandoffStatus = HandoffStatus.IN_REVIEW,
    state: ConversationState | None = None,
) -> PersistedHandoff:
    return PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key="k" * 64,
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        status=status,
        conversation_state=state
        or ConversationState(
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
        ),
    )


def test_model_is_frozen():
    review = _full_review()
    assert review.model_config.get("frozen") is True
    with pytest.raises(ValidationError):
        review.customer_phone = "+900000000000"


def test_all_intended_fields_exist():
    expected = {
        "handoff_id",
        "customer_phone",
        "customer_name",
        "reason",
        "status",
        "intent",
        "booking_stage",
        "needs_human",
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    }
    assert set(HandoffReview.model_fields) == expected


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "idempotency_key",
        "message",
        "raw_message",
        "transcript",
        "conversation_history",
        "ai_reply",
        "reply",
        "reasoning",
        "chain_of_thought",
        "system_prompt",
        "extraction_prompt",
        "prompt",
        "safety_matches",
        "safety_patterns",
        "provider",
        "provider_name",
        "model",
        "model_name",
        "api_key",
        "database_url",
        "sql",
        "repository_backend",
        "backend",
        "created_at",
        "updated_at",
    ],
)
def test_forbidden_fields_absent(forbidden_field: str):
    assert forbidden_field not in HandoffReview.model_fields


def test_enum_types_preserved():
    review = _full_review()
    assert isinstance(review.reason, HandoffReason)
    assert isinstance(review.status, HandoffStatus)
    assert isinstance(review.intent, ConversationIntent)
    assert isinstance(review.booking_stage, BookingStage)


def test_date_type_preserved():
    review = _full_review()
    assert isinstance(review.travel_date, date)
    assert review.travel_date == date(2026, 9, 10)


def test_optional_fields_allowed_none():
    review = HandoffReview(
        handoff_id=UUID("12345678123456781234567812345678"),
        customer_phone="+905551112233",
        reason=HandoffReason.SAFETY_ESCALATION,
        status=HandoffStatus.PENDING,
        intent=ConversationIntent.GENERAL_QUESTION,
        booking_stage=BookingStage.NONE,
    )
    for field in (
        "customer_name",
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    ):
        assert getattr(review, field) is None


def test_valid_full_model_values():
    review = _full_review()
    assert review.needs_human is True
    assert review.adults == 2
    assert review.children == 1
    assert review.hotel == "Korumar"


# --- Mapper ---------------------------------------------------------------------


def test_build_handoff_review_maps_all_fields():
    handoff = _persisted()
    review = build_handoff_review(handoff)

    assert review.handoff_id == handoff.id
    assert review.customer_phone == "+90555 111 2233"
    assert review.customer_name == "Mehmet Cam"
    assert review.reason is HandoffReason.BOOKING_REVIEW
    assert review.status is HandoffStatus.IN_REVIEW
    assert review.intent is ConversationIntent.BOOKING_REQUEST
    assert review.booking_stage is BookingStage.READY_FOR_REVIEW
    assert review.needs_human is True
    assert review.tour == "Ephesus tour"
    assert review.travel_date == date(2026, 9, 10)
    assert review.adults == 2
    assert review.children == 1
    assert review.cruise_ship == "Equinox"
    assert review.hotel == "Korumar"
    assert review.pickup_location == "Port"
    assert review.preferred_language == "English"


def test_mapper_does_not_mutate_input():
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    handoff = PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key="k" * 64,
        customer_phone="+905551112233",
        reason=HandoffReason.BOOKING_REVIEW,
        status=HandoffStatus.PENDING,
        conversation_state=state,
    )
    build_handoff_review(handoff)
    assert handoff.conversation_state.tour == "Ephesus tour"
    assert handoff.status is HandoffStatus.PENDING


def test_mapper_is_deterministic():
    handoff = _persisted()
    assert build_handoff_review(handoff) == build_handoff_review(handoff)


def test_summary_is_deterministic_and_human_friendly():
    summary = build_handoff_review_summary(build_handoff_review(_persisted()))
    again = build_handoff_review_summary(build_handoff_review(_persisted()))
    assert summary == again
    assert summary.startswith("Human review")
    assert "- Reason: booking_review" in summary
    assert "- Status: in_review" in summary
    assert "- Tour: Ephesus tour" in summary
    assert "- Travel date: 2026-09-10" in summary
    assert "k" * 64 not in summary
    assert "idempotency" not in summary.lower()


def test_summary_omits_unknown_fields():
    minimal = PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key="k" * 64,
        customer_phone="+905551112233",
        reason=HandoffReason.SAFETY_ESCALATION,
        status=HandoffStatus.PENDING,
        conversation_state=ConversationState(),
    )
    summary = build_handoff_review_summary(build_handoff_review(minimal))
    assert "Tour:" not in summary
    assert "Travel date:" not in summary
    assert "Customer:" not in summary

