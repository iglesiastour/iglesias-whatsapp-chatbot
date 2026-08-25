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


SAFETY_FALLBACKS: dict[SafetyCategory, str] = {
    SafetyCategory.PRICE: (
        "Our booking team will provide the latest available price based on your tour details."
    ),
    SafetyCategory.AVAILABILITY: (
        "I'll check availability with our booking team."
    ),
    SafetyCategory.BOOKING_CONFIRMATION: (
        "Your booking is not confirmed yet. Our booking team will confirm the details with you."
    ),
    SafetyCategory.CONTACT_INFORMATION: (
        "I'll have our team provide the correct contact information."
    ),
    SafetyCategory.DISCOUNT: (
        "I'll check the available options with our booking team."
    ),
    SafetyCategory.PICKUP_TIME: (
        "I'll check the pickup details with our booking team."
    ),
    SafetyCategory.GUIDE_ASSIGNMENT: (
        "Guide details will be confirmed by our booking team."
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
