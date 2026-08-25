"""Deterministic tests for the entity merge service."""

import os
from datetime import date

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.extraction import ExtractedEntities, StructuredExtraction
from app.services.entity_merge_service import merge_extraction_into_state


def extraction(**kwargs) -> StructuredExtraction:
    return StructuredExtraction(entities=ExtractedEntities(**kwargs))


def booking_request_state(**extra) -> ConversationState:
    return ConversationState(intent=ConversationIntent.BOOKING_REQUEST, **extra)


# --- Immutability ---


def test_incoming_state_not_mutated() -> None:
    state = booking_request_state(tour="Ephesus")
    snapshot = state.model_copy()
    merge_extraction_into_state(state, extraction(travel_date="2026-09-10"))
    assert state == snapshot


def test_new_object_returned() -> None:
    state = ConversationState()
    result = merge_extraction_into_state(state, extraction(tour="Ephesus"))
    assert result is not state
    assert isinstance(result, ConversationState)


# --- Merge semantics ---


def test_none_extraction_preserves_existing_values() -> None:
    state = ConversationState(
        tour="Ephesus", adults=2, hotel="Korumar Hotel"
    )
    result = merge_extraction_into_state(state, extraction())
    assert result.tour == "Ephesus"
    assert result.adults == 2
    assert result.hotel == "Korumar Hotel"


def test_non_none_extraction_overwrites_existing_values() -> None:
    state = ConversationState(adults=2, hotel="Hotel A")
    result = merge_extraction_into_state(
        state, extraction(adults=4, hotel="Hotel B")
    )
    assert result.adults == 4
    assert result.hotel == "Hotel B"


def test_adults_correction_works() -> None:
    result = merge_extraction_into_state(
        ConversationState(adults=2), extraction(adults=4)
    )
    assert result.adults == 4


def test_children_correction_works() -> None:
    result = merge_extraction_into_state(
        ConversationState(children=1), extraction(children=3)
    )
    assert result.children == 3


def test_travel_date_correction_works() -> None:
    result = merge_extraction_into_state(
        ConversationState(travel_date=date(2026, 9, 10)),
        extraction(travel_date="2026-09-20"),
    )
    assert result.travel_date == date(2026, 9, 20)


def test_strings_trimmed_before_storage() -> None:
    result = merge_extraction_into_state(
        ConversationState(), extraction(tour="  Ephesus  ")
    )
    assert result.tour == "Ephesus"


def test_blank_extracted_string_preserves_existing_value() -> None:
    result = merge_extraction_into_state(
        ConversationState(hotel="Korumar Hotel"), extraction(hotel="   ")
    )
    assert result.hotel == "Korumar Hotel"


def test_capitalization_preserved() -> None:
    result = merge_extraction_into_state(
        ConversationState(), extraction(hotel="  Korusu Boutique Otel  ")
    )
    assert result.hotel == "Korusu Boutique Otel"


def test_all_eight_entity_fields_can_merge() -> None:
    state = ConversationState()
    result = merge_extraction_into_state(
        state,
        extraction(
            tour="Ephesus",
            travel_date="2026-09-10",
            adults=2,
            children=1,
            cruise_ship="Celebrity Equinox",
            hotel="Korumar Hotel",
            pickup_location="Kusadasi Port",
            preferred_language="English",
        ),
    )
    assert result.tour == "Ephesus"
    assert result.travel_date == date(2026, 9, 10)
    assert result.adults == 2
    assert result.children == 1
    assert result.cruise_ship == "Celebrity Equinox"
    assert result.hotel == "Korumar Hotel"
    assert result.pickup_location == "Kusadasi Port"
    assert result.preferred_language == "English"


def test_no_extraction_field_deletes_existing_value() -> None:
    state = ConversationState(hotel="Korumar Hotel", pickup_location="Port")
    result = merge_extraction_into_state(state, extraction(tour="Ephesus"))
    assert result.hotel == "Korumar Hotel"
    assert result.pickup_location == "Port"


# --- Preserved conversation fields ---


def test_intent_preserved() -> None:
    state = ConversationState(intent=ConversationIntent.TOUR_INFORMATION)
    result = merge_extraction_into_state(state, extraction(tour="Ephesus"))
    assert result.intent is ConversationIntent.TOUR_INFORMATION


def test_needs_human_preserved() -> None:
    state = ConversationState(needs_human=True)
    result = merge_extraction_into_state(state, extraction(tour="Ephesus"))
    assert result.needs_human is True


# --- Booking-stage recalculation ---


def test_booking_request_incomplete_remains_collecting_details() -> None:
    state = booking_request_state(tour="Ephesus", booking_stage=BookingStage.COLLECTING_DETAILS)
    result = merge_extraction_into_state(state, extraction(adults=2))
    # travel_date still missing
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS
    assert result.adults == 2


def test_booking_request_becomes_ready_for_review_when_completed() -> None:
    state = booking_request_state(
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    # Nothing missing already; a no-op extraction keeps READY_FOR_REVIEW.
    result = merge_extraction_into_state(state, extraction())
    assert result.booking_stage is BookingStage.READY_FOR_REVIEW


def test_extraction_completing_required_fields_upgrades_stage() -> None:
    state = booking_request_state(
        tour="Ephesus", booking_stage=BookingStage.COLLECTING_DETAILS
    )
    result = merge_extraction_into_state(
        state, extraction(travel_date="2026-09-10", adults=2)
    )
    assert result.missing_booking_fields() == ()
    assert result.booking_stage is BookingStage.READY_FOR_REVIEW


def test_complete_booking_remains_ready_for_review_after_optional_update() -> None:
    state = booking_request_state(
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = merge_extraction_into_state(state, extraction(children=2))
    assert result.children == 2
    assert result.booking_stage is BookingStage.READY_FOR_REVIEW


def test_non_booking_none_stage_stays_none() -> None:
    state = ConversationState(
        intent=ConversationIntent.TOUR_INFORMATION,
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.NONE,
    )
    result = merge_extraction_into_state(state, extraction(hotel="Korumar"))
    assert result.booking_stage is BookingStage.NONE


# --- Stage protection ---


def test_human_review_remains_human_review() -> None:
    state = booking_request_state(
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.HUMAN_REVIEW,
    )
    result = merge_extraction_into_state(state, extraction(children=1))
    assert result.children == 1
    assert result.booking_stage is BookingStage.HUMAN_REVIEW


def test_confirmed_remains_confirmed_with_entity_updates() -> None:
    state = booking_request_state(
        tour="Old Tour",
        booking_stage=BookingStage.CONFIRMED,
    )
    result = merge_extraction_into_state(state, extraction(tour="New Tour", adults=5))
    assert result.tour == "New Tour"
    assert result.adults == 5
    assert result.booking_stage is BookingStage.CONFIRMED


def test_cancelled_remains_cancelled_with_entity_updates() -> None:
    state = booking_request_state(booking_stage=BookingStage.CANCELLED)
    result = merge_extraction_into_state(state, extraction(travel_date="2026-10-01"))
    assert result.travel_date == date(2026, 10, 1)
    assert result.booking_stage is BookingStage.CANCELLED


# --- Determinism / environment ---


def test_repeated_same_merge_is_deterministic() -> None:
    state = booking_request_state(tour="Ephesus")
    first = merge_extraction_into_state(state, extraction(adults=2))
    second = merge_extraction_into_state(state, extraction(adults=2))
    assert first == second


def test_no_environment_dependency() -> None:
    import os

    snapshot = dict(os.environ)
    merge_extraction_into_state(
        ConversationState(), extraction(tour="Ephesus")
    )
    assert dict(os.environ) == snapshot

