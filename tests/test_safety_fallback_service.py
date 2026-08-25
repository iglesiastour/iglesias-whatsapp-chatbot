"""Deterministic tests for the contextual safety fallback builder (no network)."""

import os
from datetime import date

import pytest

from app.models.conversation import BookingStage
from app.prompts.policies import SafetyCategory, get_safety_fallback
from app.services.safety_fallback_service import (
    SafetyFallbackContext,
    build_contextual_safety_fallback,
)


def test_no_state_returns_existing_fallback() -> None:
    for category in SafetyCategory:
        result = build_contextual_safety_fallback(category, ctx=None)
        assert result == get_safety_fallback(category)


def test_ready_for_review_with_all_fields() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "September 10, 2026" in result
    assert "2 adults" in result
    assert "Our team can review the request and confirm the next steps" in result


def test_ready_for_review_does_not_ask_children() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "children" not in result.lower()


def test_ready_for_review_does_not_ask_hotel() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        hotel="Korumar Hotel",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "hotel" not in result.lower()


def test_ready_for_review_does_not_ask_cruise_ship() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        cruise_ship="Celebrity Equinox",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "cruise" not in result.lower()


def test_ready_for_review_contains_no_price_claim() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "€" not in result
    assert "$" not in result
    assert "£" not in result
    assert "price" not in result.lower()


def test_ready_for_review_contains_no_availability_claim() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "available" not in result.lower()
    assert "availability" not in result.lower()


def test_ready_for_review_contains_no_operational_promise() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "forward" not in result.lower()
    assert "in touch" not in result.lower()
    assert "callback" not in result.lower()


def test_ready_for_review_only_uses_verified_values() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "September 10, 2026" in result
    assert "2 adults" in result


def test_partial_ready_state_tour_only() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "Our team can review the request and confirm the next steps" in result


def test_partial_ready_state_date_and_adults() -> None:
    ctx = SafetyFallbackContext(
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "September 10, 2026" in result
    assert "2 adults" in result
    assert "Our team can review the request and confirm the next steps" in result


def test_collecting_details_asks_only_missing_required_fields() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        booking_stage=BookingStage.COLLECTING_DETAILS,
        missing_booking_fields=("travel_date", "adults"),
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "travel date" in result.lower()
    assert "number of adults" in result.lower()
    assert "children" not in result.lower()
    assert "hotel" not in result.lower()


def test_collecting_details_uses_missing_booking_fields() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        booking_stage=BookingStage.COLLECTING_DETAILS,
        missing_booking_fields=("adults",),
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "September 10, 2026" in result
    assert "number of adults" in result.lower()


def test_human_review_stays_non_committal() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.HUMAN_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "requires review by our team" in result
    assert "forward" not in result.lower()
    assert "callback" not in result.lower()


def test_human_review_via_requires_human_flag() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        requires_human=True,
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "requires review by our team" in result


def test_confirmed_state_acknowledges_status() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.CONFIRMED,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "marked as confirmed" in result


def test_cancelled_state_acknowledges_status() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.CANCELLED,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "marked as cancelled" in result


def test_price_category_retains_safe_specificity() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(SafetyCategory.PRICE, ctx=ctx)
    assert result == get_safety_fallback(SafetyCategory.PRICE)


def test_availability_category_retains_safe_specificity() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(SafetyCategory.AVAILABILITY, ctx=ctx)
    assert result == get_safety_fallback(SafetyCategory.AVAILABILITY)


def test_deterministic_repeated_calls() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result1 = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    result2 = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert result1 == result2


def test_customer_phone_impossible_to_leak() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "+9055" not in result
    assert "phone" not in result.lower()


def test_no_network_env_dependency() -> None:
    snapshot = dict(os.environ)
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    build_contextual_safety_fallback(SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx)
    assert dict(os.environ) == snapshot


def test_single_adult_singular_form() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=1,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "1 adult" in result
    assert "1 adults" not in result


def test_multiple_adults_plural_form() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=3,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "3 adults" in result


def test_ready_for_review_empty_state_falls_back() -> None:
    ctx = SafetyFallbackContext(
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert result == get_safety_fallback(SafetyCategory.OPERATIONAL_PROMISE)


def test_collecting_details_empty_known_falls_back() -> None:
    ctx = SafetyFallbackContext(
        booking_stage=BookingStage.COLLECTING_DETAILS,
        missing_booking_fields=("tour", "travel_date", "adults"),
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPERATIONAL_PROMISE, ctx=ctx
    )
    assert "tour" in result.lower()
    assert "travel date" in result.lower()
    assert "number of adults" in result.lower()


def test_unsupported_detail_with_ready_for_review_uses_context() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.UNSUPPORTED_DETAIL, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "Our team can review the request and confirm the next steps" in result


def test_optional_field_reask_with_ready_for_review_uses_context() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.OPTIONAL_FIELD_REASK, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "Our team can review the request and confirm the next steps" in result


def test_unsupported_detail_with_collecting_details_uses_context() -> None:
    ctx = SafetyFallbackContext(
        tour="Ephesus tour",
        booking_stage=BookingStage.COLLECTING_DETAILS,
        missing_booking_fields=("travel_date", "adults"),
    )
    result = build_contextual_safety_fallback(
        SafetyCategory.UNSUPPORTED_DETAIL, ctx=ctx
    )
    assert "Ephesus tour" in result
    assert "travel date" in result.lower()
    assert "number of adults" in result.lower()
