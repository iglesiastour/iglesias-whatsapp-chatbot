"""Deterministic tests for the output safety validator (no network, no env)."""

import dataclasses
import os

import pytest

from app.prompts.policies import SafetyCategory, get_safety_fallback
from app.security.output_guard import OutputSafetyResult, inspect_ai_output


SAFE_OUTPUTS = (
    "",
    "   ",
    "Our Ephesus tours are guided by licensed professionals.",
    "I'd be happy to help you plan your visit to Ephesus!",
    "Could you tell me your travel dates and group size?",
    "Our booking team will provide the latest available price.",
    "I cannot provide a confirmed price yet.",
    "Please tell me your group size so our team can quote you.",
    "I'll check the pickup time with our booking team.",
    "Pickup details will be confirmed later.",
)


def test_empty_output_is_safe() -> None:
    assert inspect_ai_output("").is_safe


def test_whitespace_output_is_safe() -> None:
    result = inspect_ai_output("   ")
    assert result.is_safe
    assert result.violations == ()


@pytest.mark.parametrize("text", SAFE_OUTPUTS)
def test_normal_outputs_are_safe(text: str) -> None:
    result = inspect_ai_output(text)
    assert result.is_safe
    assert result.violations == ()


@pytest.mark.parametrize("category", list(SafetyCategory))
def test_all_safety_fallback_strings_are_safe(category: SafetyCategory) -> None:
    assert inspect_ai_output(get_safety_fallback(category)).is_safe is True


# --- BOOKING_CONFIRMATION ---


def test_booking_confirmation_detected() -> None:
    result = inspect_ai_output("Your booking is confirmed for tomorrow.")
    assert not result.is_safe
    assert SafetyCategory.BOOKING_CONFIRMATION in result.violations


def test_booking_negation_is_safe() -> None:
    for text in (
        "Your booking is not confirmed yet.",
        "Our booking team will confirm the details with you.",
        "I cannot confirm the booking yet.",
        "I'll ask our team to confirm it.",
    ):
        assert SafetyCategory.BOOKING_CONFIRMATION not in inspect_ai_output(text).violations


# --- AVAILABILITY ---


def test_availability_claim_detected() -> None:
    result = inspect_ai_output("Yes, we have availability for tomorrow.")
    assert SafetyCategory.AVAILABILITY in result.violations
    assert not result.is_safe


def test_availability_fallback_is_safe() -> None:
    result = inspect_ai_output("I'll check availability with our booking team.")
    assert result.is_safe
    assert SafetyCategory.AVAILABILITY not in result.violations


def test_availability_negation_is_safe() -> None:
    assert inspect_ai_output("I cannot confirm availability yet.").is_safe
    assert inspect_ai_output("Availability needs to be checked.").is_safe


# --- PRICE ---


def test_euro_price_detected() -> None:
    assert SafetyCategory.PRICE in inspect_ai_output("The tour costs €75 per person.").violations


def test_usd_price_detected() -> None:
    assert SafetyCategory.PRICE in inspect_ai_output("The price is 120 USD.").violations


def test_dollar_price_detected() -> None:
    assert SafetyCategory.PRICE in inspect_ai_output("It will be $90 per person.").violations


def test_eur_textual_price_detected() -> None:
    assert SafetyCategory.PRICE in inspect_ai_output("The total price is 150 EUR.").violations


def test_euros_word_price_detected() -> None:
    assert SafetyCategory.PRICE in inspect_ai_output("The tour is 80 euros per person.").violations


# --- DISCOUNT ---


def test_discount_promise_detected() -> None:
    result = inspect_ai_output("I can give you a 10% discount.")
    assert SafetyCategory.DISCOUNT in result.violations


def test_percent_off_detected() -> None:
    assert SafetyCategory.DISCOUNT in inspect_ai_output("We can offer 15% off.").violations


def test_discount_fallback_is_safe() -> None:
    result = inspect_ai_output(
        "I'll check available discount options with our booking team."
    )
    assert result.is_safe
    assert SafetyCategory.DISCOUNT not in result.violations


# --- PICKUP_TIME ---


def test_pickup_time_claim_detected() -> None:
    result = inspect_ai_output("Your pickup time is 8:30 AM.")
    assert SafetyCategory.PICKUP_TIME in result.violations


def test_pickup_confirmation_detected() -> None:
    result = inspect_ai_output("We will pick you up at 09:00 from your hotel.")
    assert SafetyCategory.PICKUP_TIME in result.violations


def test_pickup_driver_arrival_detected() -> None:
    result = inspect_ai_output("Your driver will arrive at 08:15.")
    assert SafetyCategory.PICKUP_TIME in result.violations


def test_pickup_fallback_is_safe() -> None:
    result = inspect_ai_output("I'll check the pickup details with our booking team.")
    assert result.is_safe
    assert SafetyCategory.PICKUP_TIME not in result.violations


# --- CONTACT_INFORMATION ---


def test_phone_contact_claim_detected() -> None:
    result = inspect_ai_output("Call us at +90 212 555 1234.")
    assert SafetyCategory.CONTACT_INFORMATION in result.violations


def test_whatsapp_contact_claim_detected() -> None:
    result = inspect_ai_output("Our WhatsApp number is +90 555 123 4567.")
    assert SafetyCategory.CONTACT_INFORMATION in result.violations


def test_email_contact_claim_detected() -> None:
    result = inspect_ai_output("Email us at booking@example.com.")
    assert SafetyCategory.CONTACT_INFORMATION in result.violations


def test_our_email_contact_claim_detected() -> None:
    result = inspect_ai_output("Our email is info@example.com.")
    assert SafetyCategory.CONTACT_INFORMATION in result.violations


# --- MULTIPLE VIOLATIONS ---


def test_multiple_categories_detected_together() -> None:
    result = inspect_ai_output(
        "Your booking is confirmed for tomorrow at 09:00. The total price is €150."
    )
    assert result.violations == (
        SafetyCategory.BOOKING_CONFIRMATION,
        SafetyCategory.PICKUP_TIME,
        SafetyCategory.PRICE,
    )


def test_no_duplicate_categories() -> None:
    result = inspect_ai_output("The tour costs €75. The total price is €150. That is 75 EUR.")
    assert result.violations.count(SafetyCategory.PRICE) == 1
    assert len(result.violations) == len(set(result.violations))


def test_violation_order_is_deterministic() -> None:
    first = inspect_ai_output("The price is 120 USD. We have seats available.")
    second = inspect_ai_output("We have seats available. The price is 120 USD.")
    assert first.violations == second.violations


# --- MECHANICS ---


def test_result_object_is_immutable() -> None:
    result = inspect_ai_output("The tour costs €75.")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.is_safe = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.violations = ()  # type: ignore[misc]


def test_case_insensitive_matching() -> None:
    assert not inspect_ai_output("YOUR BOOKING IS CONFIRMED.").is_safe
    assert not inspect_ai_output("the TOUR COSTS €75").is_safe


def test_whitespace_normalization() -> None:
    result = inspect_ai_output("Your   booking   is\n\tconfirmed.")
    assert SafetyCategory.BOOKING_CONFIRMATION in result.violations


def test_original_text_is_not_mutated() -> None:
    original = "  The TOUR costs €75.  "
    inspect_ai_output(original)
    assert original == "  The TOUR costs €75.  "


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    inspect_ai_output("Your booking is confirmed.")
    assert dict(os.environ) == snapshot

