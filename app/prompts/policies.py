"""Centralized AI safety policy primitives for Iglesias Tour Turkey.

These are deterministic, application-owned safety rules. They are the single
source of truth for safe fallback responses and for the categories of facts
the AI must never state without verification.
"""

from enum import StrEnum


class SafetyCategory(StrEnum):
    """Categories of operational facts that require human/booking-team verification."""

    PRICE = "price"
    AVAILABILITY = "availability"
    BOOKING_CONFIRMATION = "booking_confirmation"
    CONTACT_INFORMATION = "contact_information"
    DISCOUNT = "discount"
    PICKUP_TIME = "pickup_time"
    GUIDE_ASSIGNMENT = "guide_assignment"
    OPERATIONAL_PROMISE = "operational_promise"
    UNSUPPORTED_DETAIL = "unsupported_detail"
    OPTIONAL_FIELD_REASK = "optional_field_reask"


SAFETY_FALLBACKS: dict[SafetyCategory, str] = {
    SafetyCategory.PRICE: (
        "Our booking team will provide the latest available price based on your tour details."
    ),
    SafetyCategory.AVAILABILITY: (
        "Availability needs to be confirmed by our booking team."
    ),
    SafetyCategory.BOOKING_CONFIRMATION: (
        "Your booking is not confirmed yet. Our booking team will confirm the details with you."
    ),
    SafetyCategory.CONTACT_INFORMATION: (
        "Our team can provide the correct contact information."
    ),
    SafetyCategory.DISCOUNT: (
        "Our booking team can check the available discount options."
    ),
    SafetyCategory.PICKUP_TIME: (
        "Pickup details will be confirmed by our booking team."
    ),
    SafetyCategory.GUIDE_ASSIGNMENT: (
        "Guide details will be confirmed by our booking team."
    ),
    SafetyCategory.OPERATIONAL_PROMISE: (
        "I have the details you've provided. Our team can review the request and confirm the next steps."
    ),
    SafetyCategory.UNSUPPORTED_DETAIL: (
        "I can help using the tour details currently available. Our team can confirm any additional options or specifics."
    ),
    SafetyCategory.OPTIONAL_FIELD_REASK: (
        "I have the required booking details noted. Our team can review the request and confirm the next steps."
    ),
}


FORBIDDEN_UNVERIFIED_FACTS: tuple[str, ...] = (
    "phone numbers",
    "email addresses",
    "WhatsApp numbers",
    "prices",
    "discounts",
    "availability",
    "booking confirmations",
    "pickup times",
    "hotel names",
    "guide names",
    "vehicle assignments",
    "payment confirmations",
)


def get_safety_fallback(category: SafetyCategory) -> str:
    """Return the safe fallback response for a safety category.

    Raises:
        KeyError: If the category has no defined fallback. This is intentional
            so programming errors surface instead of being silently hidden.
    """
    return SAFETY_FALLBACKS[category]
