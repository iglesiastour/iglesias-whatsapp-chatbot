"""Deterministic tests for the output safety validator (no network, no env)."""

import dataclasses
import os

import pytest

from app.models.conversation import BookingStage
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
    result = inspect_ai_output("Availability needs to be confirmed by our booking team.")
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


# --- OPERATIONAL_PROMISE ---


def test_forward_to_booking_team_detected() -> None:
    result = inspect_ai_output("I'll forward this to our booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_theyll_be_in_touch_detected() -> None:
    result = inspect_ai_output("They'll be in touch shortly.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_i_can_check_availability_detected() -> None:
    result = inspect_ai_output("I can check availability for you.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_will_send_exact_price_detected() -> None:
    result = inspect_ai_output("We'll send you the exact price.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_forward_to_team_detected() -> None:
    result = inspect_ai_output("I'll forward this to the team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_pass_along_to_team_detected() -> None:
    result = inspect_ai_output("I'll pass this along to our team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_will_review_request_detected() -> None:
    result = inspect_ai_output("We will review your request.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_operational_promise_case_insensitive() -> None:
    result = inspect_ai_output("I'LL FORWARD THIS TO OUR BOOKING TEAM.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


# --- Safe non-commitment language ---


def test_our_team_can_review_is_safe() -> None:
    result = inspect_ai_output("Our team can review the request.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_availability_needs_confirmation_is_safe() -> None:
    result = inspect_ai_output("Availability needs to be confirmed by our team.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_next_step_is_review_is_safe() -> None:
    result = inspect_ai_output("The next step is review by our booking team.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_availability_pricing_confirmed_by_team_is_safe() -> None:
    result = inspect_ai_output(
        "Our booking team can confirm availability and pricing."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


# --- UNSUPPORTED_DETAIL ---


def test_private_ephesus_tour_unsupported() -> None:
    result = inspect_ai_output(
        "The private Ephesus tour includes a licensed guide.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL in result.violations


def test_luxury_ephesus_tour_unsupported() -> None:
    result = inspect_ai_output(
        "The luxury Ephesus tour is very popular.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL in result.violations


def test_vip_ephesus_tour_unsupported() -> None:
    result = inspect_ai_output(
        "The VIP Ephesus tour offers premium service.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL in result.violations


def test_biblical_ephesus_tour_unsupported() -> None:
    result = inspect_ai_output(
        "Our Biblical Ephesus tour covers all sites.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL in result.violations


def test_bare_ephesus_tour_is_safe() -> None:
    result = inspect_ai_output(
        "The Ephesus tour includes a licensed guide.",
        known_tour="Ephesus tour",
    )
    assert result.is_safe
    assert SafetyCategory.UNSUPPORTED_DETAIL not in result.violations


def test_your_ephesus_tour_is_safe() -> None:
    result = inspect_ai_output(
        "Your Ephesus tour is scheduled for tomorrow.",
        known_tour="Ephesus tour",
    )
    assert result.is_safe
    assert SafetyCategory.UNSUPPORTED_DETAIL not in result.violations


def test_the_ephesus_tour_is_safe() -> None:
    result = inspect_ai_output(
        "The Ephesus tour is very popular.",
        known_tour="Ephesus tour",
    )
    assert result.is_safe
    assert SafetyCategory.UNSUPPORTED_DETAIL not in result.violations


def test_none_known_tour_skips_unsupported_check() -> None:
    result = inspect_ai_output(
        "The private Ephesus tour includes a licensed guide.",
        known_tour=None,
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL not in result.violations


def test_known_tour_not_in_text_no_violation() -> None:
    result = inspect_ai_output(
        "The Pamukkale tour is wonderful.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.UNSUPPORTED_DETAIL not in result.violations


def test_multiple_violations_deterministic_order() -> None:
    result = inspect_ai_output(
        "I'll forward this to our booking team. The tour costs €75.",
        known_tour="Ephesus tour",
    )
    assert SafetyCategory.PRICE in result.violations
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_existing_safety_fallbacks_still_safe() -> None:
    for category in list(SafetyCategory):
        fallback = get_safety_fallback(category)
        result = inspect_ai_output(fallback, known_tour="Ephesus tour")
        assert result.is_safe is True


# --- OPERATIONAL_PROMISE pattern hardening ---


def test_check_with_booking_team_detected() -> None:
    result = inspect_ai_output("I'll check this with our booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_curly_apostrophe_check_with_booking_team_detected() -> None:
    """Curly apostrophe in I\u2019ll must still trigger detection."""
    result = inspect_ai_output("I\u2019ll check this with our booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_well_check_with_booking_team_detected() -> None:
    result = inspect_ai_output("We'll check this with our booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_forward_request_to_booking_team_detected() -> None:
    result = inspect_ai_output("I'll forward your request to our booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_pass_details_to_team_detected() -> None:
    result = inspect_ai_output("We'll pass your details to our team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_i_will_send_to_booking_team_detected() -> None:
    result = inspect_ai_output("I will send this to the booking team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_get_back_to_you_shortly_detected() -> None:
    result = inspect_ai_output("I'll get back to you shortly.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_well_get_back_to_you_soon_detected() -> None:
    result = inspect_ai_output("We'll get back to you soon.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_contact_you_shortly_detected() -> None:
    result = inspect_ai_output("Our team will contact you shortly.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_be_in_touch_shortly_detected() -> None:
    result = inspect_ai_output("They'll be in touch shortly.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_exact_live_escape_sentence_detected() -> None:
    """The exact sentence that escaped in live testing must trigger."""
    result = inspect_ai_output(
        "Thank you for confirming the details. I have noted your Ephesus tour "
        "for September 10, 2026 for 2 adults. I'll check this with our booking "
        "team and get back to you shortly."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_curly_apostrophe_well_get_back_detected() -> None:
    result = inspect_ai_output("We\u2019ll get back to you shortly.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_i_will_get_back_to_you_detected() -> None:
    result = inspect_ai_output("I will get back to you soon.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


# --- Safe language preservation (must NOT trigger) ---


def test_our_team_can_review_still_safe() -> None:
    result = inspect_ai_output("Our team can review the request.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_can_confirm_availability_still_safe() -> None:
    result = inspect_ai_output(
        "Our booking team can confirm availability and pricing."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_needs_to_be_confirmed_still_safe() -> None:
    result = inspect_ai_output(
        "Availability needs to be confirmed by our team."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_ill_help_you_still_safe() -> None:
    result = inspect_ai_output(
        "I'll help you with the information I have."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_you_can_contact_team_still_safe() -> None:
    result = inspect_ai_output("You can contact our team for assistance.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_may_need_info_still_safe() -> None:
    result = inspect_ai_output("Our team may need additional information.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


# --- OPTIONAL_FIELD_REASK ---


def test_children_question_detected_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "How many children will be joining the tour?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations
    assert not result.is_safe


def test_hotel_question_detected_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "Which hotel are you staying at?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_cruise_ship_question_detected_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "Which cruise ship are you arriving on?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_pickup_location_question_detected_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "Where are you staying for pickup?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_language_question_detected_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "What is your preferred language?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_optional_field_question_not_detected_in_collecting_details() -> None:
    result = inspect_ai_output(
        "How many children will be joining the tour?",
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK not in result.violations


def test_optional_field_question_not_detected_when_no_stage() -> None:
    result = inspect_ai_output(
        "How many children will be joining the tour?",
        booking_stage=None,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK not in result.violations


def test_safe_reply_not_flagged_in_ready_for_review() -> None:
    result = inspect_ai_output(
        "I have the required booking details noted. Our team can review the request and confirm the next steps.",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert result.is_safe
    assert SafetyCategory.OPTIONAL_FIELD_REASK not in result.violations


def test_optional_field_reask_fallback_is_safe() -> None:
    fallback = get_safety_fallback(SafetyCategory.OPTIONAL_FIELD_REASK)
    result = inspect_ai_output(
        fallback,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert result.is_safe


def test_children_count_question_detected() -> None:
    result = inspect_ai_output(
        "Number of children attending?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_hotel_name_question_detected() -> None:
    result = inspect_ai_output(
        "What is your hotel name?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


def test_language_preference_question_detected() -> None:
    result = inspect_ai_output(
        "Do you prefer English or Turkish?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK not in result.violations


def test_curly_apostrophe_optional_field_detected() -> None:
    """Curly apostrophe in 'What\u2019s your hotel?' must still trigger detection."""
    result = inspect_ai_output(
        "What\u2019s your hotel name?",
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    assert SafetyCategory.OPTIONAL_FIELD_REASK in result.violations


# --- OPERATIONAL_PROMISE: check availability/pricing with team ---


def test_check_availability_and_pricing_with_booking_team_detected() -> None:
    result = inspect_ai_output(
        "I'll check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_curly_apostrophe_check_availability_and_pricing_detected() -> None:
    """Curly apostrophe in I\u2019ll must still trigger detection."""
    result = inspect_ai_output(
        "I\u2019ll check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_well_check_availability_and_pricing_detected() -> None:
    result = inspect_ai_output(
        "We'll check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_i_will_check_availability_and_pricing_detected() -> None:
    result = inspect_ai_output(
        "I will check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_check_pricing_and_availability_detected() -> None:
    result = inspect_ai_output(
        "I'll check pricing and availability with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_check_price_with_team_detected() -> None:
    result = inspect_ai_output("I'll check the price with our team.")
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_check_availability_with_team_and_let_you_know_detected() -> None:
    result = inspect_ai_output(
        "I'll check availability with our booking team and let you know."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_once_i_have_those_details_check_detected() -> None:
    """Exact live escape sentence must trigger."""
    result = inspect_ai_output(
        "Once I have those details, I'll check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_once_i_have_those_details_curly_apostrophe_detected() -> None:
    """Exact live escape sentence with curly apostrophe must trigger."""
    result = inspect_ai_output(
        "Once I have those details, I\u2019ll check availability and pricing with our booking team."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_check_availability_and_get_back_detected() -> None:
    result = inspect_ai_output(
        "I'll check availability and get back to you."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


def test_check_pricing_and_contact_you_detected() -> None:
    result = inspect_ai_output(
        "We'll check pricing and contact you."
    )
    assert SafetyCategory.OPERATIONAL_PROMISE in result.violations


# --- Safe non-committal variants (must NOT trigger) ---


def test_availability_needs_confirmation_still_safe() -> None:
    result = inspect_ai_output(
        "Availability needs to be confirmed by our team."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_pricing_needs_confirmation_still_safe() -> None:
    result = inspect_ai_output(
        "Pricing needs to be confirmed by our booking team."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_booking_team_can_confirm_availability_still_safe() -> None:
    result = inspect_ai_output(
        "Our booking team can confirm availability and pricing."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_team_can_review_still_safe() -> None:
    result = inspect_ai_output("Our team can review the request.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_next_step_is_review_still_safe() -> None:
    result = inspect_ai_output(
        "The next step is review by our booking team."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_availability_pricing_require_confirmation_still_safe() -> None:
    result = inspect_ai_output(
        "Availability and pricing require confirmation by our team."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_ill_help_you_with_information_still_safe() -> None:
    result = inspect_ai_output(
        "I'll help you with the information I have."
    )
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations


def test_you_can_contact_team_for_assistance_still_safe() -> None:
    result = inspect_ai_output("You can contact our team for assistance.")
    assert result.is_safe
    assert SafetyCategory.OPERATIONAL_PROMISE not in result.violations

