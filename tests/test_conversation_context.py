"""Deterministic tests for the conversation-state reply-context builder."""

import os
from datetime import date

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.prompts.conversation_context import build_conversation_context


def test_default_state_returns_non_empty_context() -> None:
    context = build_conversation_context(ConversationState())
    assert context.strip() != ""


def _full_state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Celebrity Equinox",
        hotel="Korumar Hotel",
        pickup_location="Kusadasi Port",
        preferred_language="English",
        booking_stage=BookingStage.COLLECTING_DETAILS,
        needs_human=False,
    )


def test_unknown_fields_omitted() -> None:
    context = build_conversation_context(ConversationState())
    assert "Tour: None" not in context
    assert "- Tour:" not in context
    assert "- Travel date:" not in context


def test_known_tour_included() -> None:
    context = build_conversation_context(ConversationState(tour="Ephesus"))
    assert "- Tour: Ephesus" in context


def test_travel_date_included_iso() -> None:
    context = build_conversation_context(
        ConversationState(travel_date=date(2026, 9, 10))
    )
    assert "- Travel date: 2026-09-10" in context


def test_adults_included() -> None:
    context = build_conversation_context(ConversationState(adults=2))
    assert "- Adults: 2" in context


def test_children_included() -> None:
    context = build_conversation_context(ConversationState(children=1))
    assert "- Children: 1" in context


def test_cruise_ship_included() -> None:
    context = build_conversation_context(
        ConversationState(cruise_ship="Celebrity Equinox")
    )
    assert "- Cruise ship: Celebrity Equinox" in context


def test_hotel_included() -> None:
    context = build_conversation_context(ConversationState(hotel="Korumar Hotel"))
    assert "- Hotel: Korumar Hotel" in context


def test_pickup_location_included() -> None:
    context = build_conversation_context(
        ConversationState(pickup_location="Kusadasi Port")
    )
    assert "- Pickup location: Kusadasi Port" in context


def test_preferred_language_included() -> None:
    context = build_conversation_context(
        ConversationState(preferred_language="English")
    )
    assert "- Preferred language: English" in context


def test_intent_included() -> None:
    context = build_conversation_context(
        ConversationState(intent=ConversationIntent.BOOKING_REQUEST)
    )
    assert "Current intent: booking_request" in context


def test_booking_stage_included() -> None:
    context = build_conversation_context(
        ConversationState(booking_stage=BookingStage.READY_FOR_REVIEW)
    )
    assert "Booking stage: ready_for_review" in context


# --- Stage-specific behavior ---


def test_ready_for_review_says_required_details_complete() -> None:
    context = build_conversation_context(
        ConversationState(
            intent=ConversationIntent.BOOKING_REQUEST,
            booking_stage=BookingStage.READY_FOR_REVIEW,
        )
    )
    assert "required booking details already known are complete" in context.lower()


def test_ready_for_review_says_do_not_reask_known_fields() -> None:
    context = build_conversation_context(
        ConversationState(
            intent=ConversationIntent.BOOKING_REQUEST,
            tour="Ephesus",
            travel_date=date(2026, 9, 10),
            adults=2,
            booking_stage=BookingStage.READY_FOR_REVIEW,
        )
    )
    assert "Do not ask again for tour, travel date, or adults" in context


def test_ready_for_review_says_do_not_ask_optional_fields() -> None:
    context = build_conversation_context(
        ConversationState(
            intent=ConversationIntent.BOOKING_REQUEST,
            tour="Ephesus",
            travel_date=date(2026, 9, 10),
            adults=2,
            booking_stage=BookingStage.READY_FOR_REVIEW,
        )
    )
    assert "Do not ask for additional optional details such as children, cruise ship, hotel, pickup location, or preferred language" in context


def test_collecting_details_includes_only_missing_fields() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    context = build_conversation_context(state)
    section = context.split("Still needed for booking:")[1].split(
        "Ask only for missing required booking information."
    )[0]
    assert "- adults" in section
    assert "- travel date" not in section
    assert "- tour" not in section


def test_collecting_details_uses_missing_booking_fields() -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    expected = set(state.missing_booking_fields())
    context = build_conversation_context(state)
    section = context.split("Still needed for booking:")[1].split(
        "Ask only for missing required booking information."
    )[0]
    shown = {line.strip("- ").strip() for line in section.strip().splitlines()}
    shown_normalized = {
        "travel_date" if f == "travel date" else f for f in shown
    }
    assert shown_normalized == expected


def test_collecting_details_ask_only_missing_instruction() -> None:
    context = build_conversation_context(
        ConversationState(booking_stage=BookingStage.COLLECTING_DETAILS)
    )
    assert "Ask only for missing required booking information" in context


def test_human_review_instructs_human_follow_up() -> None:
    context = build_conversation_context(
        ConversationState(booking_stage=BookingStage.HUMAN_REVIEW)
    )
    assert "This conversation requires human follow-up" in context


def test_requires_human_flag_triggers_human_follow_up() -> None:
    context = build_conversation_context(ConversationState(needs_human=True))
    assert "This conversation requires human follow-up" in context


def test_confirmed_marked_backend_confirmed() -> None:
    context = build_conversation_context(
        ConversationState(booking_stage=BookingStage.CONFIRMED)
    )
    assert "Backend status: confirmed" in context


def test_cancelled_marked_backend_cancelled() -> None:
    context = build_conversation_context(
        ConversationState(booking_stage=BookingStage.CANCELLED)
    )
    assert "Backend status: cancelled" in context


# --- Safety / hygiene ---


def test_safety_instruction_always_present() -> None:
    for state in (ConversationState(), _full_state()):
        assert "Do not invent price, availability, contact details" in (
            build_conversation_context(state)
        )


def test_customer_phone_never_present() -> None:
    context = build_conversation_context(_full_state())
    assert "+9055" not in context
    assert "customer_phone" not in context


def test_no_raw_state_repr_or_json() -> None:
    context = build_conversation_context(_full_state())
    assert "ConversationState(" not in context
    assert "{" not in context


def test_no_none_placeholder() -> None:
    context = build_conversation_context(ConversationState())
    assert "None" not in context


def test_deterministic_repeated_calls() -> None:
    for state in (ConversationState(), _full_state()):
        assert build_conversation_context(state) == build_conversation_context(state)


def test_no_network_or_environment_dependency() -> None:
    snapshot = dict(os.environ)
    build_conversation_context(_full_state())
    assert dict(os.environ) == snapshot


# --- Customer name behavior ---


def test_known_name_included() -> None:
    context = build_conversation_context(ConversationState(), customer_name="Maria")
    assert "- Customer name: Maria" in context


def test_name_whitespace_normalized() -> None:
    context = build_conversation_context(
        ConversationState(), customer_name="  Maria   Lopez  "
    )
    assert "- Customer name: Maria Lopez" in context


def test_name_capitalization_preserved() -> None:
    context = build_conversation_context(
        ConversationState(), customer_name="  maria   LOPEZ  "
    )
    assert "- Customer name: maria LOPEZ" in context


def test_blank_name_treated_unknown() -> None:
    context = build_conversation_context(ConversationState(), customer_name="   ")
    assert "Customer name is not known" in context
    assert "- Customer name:" not in context


def test_none_name_treated_unknown() -> None:
    context = build_conversation_context(ConversationState(), customer_name=None)
    assert "Customer name is not known" in context
    assert "- Customer name:" not in context


def test_unknown_name_no_invention_instruction() -> None:
    context = build_conversation_context(ConversationState())
    assert (
        "Do not invent, guess, infer, or use a customer name." in context
    )


def test_known_name_do_not_use_other_instruction() -> None:
    context = build_conversation_context(ConversationState(), customer_name="Maria")
    assert "Do not use any other customer name." in context


def test_name_does_not_alter_booking_state_details() -> None:
    state = _full_state()
    with_name = build_conversation_context(state, customer_name="Maria")
    without_name = build_conversation_context(state)
    assert "- Tour: Ephesus" in with_name
    assert "- Travel date: 2026-09-10" in with_name
    assert "- Adults: 2" in with_name
    assert "- Tour: Ephesus" in without_name
    assert "- Travel date: 2026-09-10" in without_name
    assert "- Adults: 2" in without_name


def test_customer_phone_still_never_present_with_name() -> None:
    context = build_conversation_context(_full_state(), customer_name="Maria")
    assert "+9055" not in context
    assert "customer_phone" not in context


def test_deterministic_repeated_calls_with_name() -> None:
    state = _full_state()
    ctx1 = build_conversation_context(state, customer_name="Maria")
    ctx2 = build_conversation_context(state, customer_name="Maria")
    assert ctx1 == ctx2