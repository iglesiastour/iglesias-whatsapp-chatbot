"""Deterministic contextual safety fallback builder.

Provides state-aware fallback responses when output guard blocks AI replies.
Pure deterministic function only: no network, no repository, no provider.
"""

from dataclasses import dataclass
from datetime import date

from app.models.conversation import BookingStage
from app.prompts.policies import SafetyCategory, get_safety_fallback

_MONTH_NAMES: tuple[str, ...] = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


@dataclass(frozen=True)
class SafetyFallbackContext:
    """Immutable context for building contextual safety fallbacks."""

    tour: str | None = None
    travel_date: date | None = None
    adults: int | None = None
    children: int | None = None
    cruise_ship: str | None = None
    hotel: str | None = None
    pickup_location: str | None = None
    preferred_language: str | None = None
    booking_stage: BookingStage = BookingStage.NONE
    requires_human: bool = False
    missing_booking_fields: tuple[str, ...] = ()


def _format_date_human_readable(d: date) -> str:
    """Format date deterministically without locale dependencies."""
    return f"{_MONTH_NAMES[d.month - 1]} {d.day}, {d.year}"


def _build_ready_for_review_fallback(ctx: SafetyFallbackContext) -> str:
    """Build contextual fallback for READY_FOR_REVIEW stage."""
    parts: list[str] = []

    if ctx.tour:
        parts.append(ctx.tour)
    if ctx.travel_date:
        parts.append(_format_date_human_readable(ctx.travel_date))
    if ctx.adults is not None:
        parts.append(f"{ctx.adults} adult{'s' if ctx.adults != 1 else ''}")

    if parts:
        detail_str = " for ".join(
            [parts[0]] + [", ".join(parts[1:])] if len(parts) > 1 else parts
        )
        return (
            f"I have your {detail_str} request noted. "
            "Our team can review the request and confirm the next steps."
        )

    return get_safety_fallback(SafetyCategory.OPERATIONAL_PROMISE)


def _build_collecting_details_fallback(ctx: SafetyFallbackContext) -> str:
    """Build contextual fallback for COLLECTING_DETAILS stage."""
    known_parts: list[str] = []

    if ctx.tour:
        known_parts.append(ctx.tour)
    if ctx.travel_date:
        known_parts.append(_format_date_human_readable(ctx.travel_date))
    if ctx.adults is not None:
        known_parts.append(f"{ctx.adults} adult{'s' if ctx.adults != 1 else ''}")

    missing_labels: list[str] = []
    _FIELD_LABELS: dict[str, str] = {
        "tour": "tour",
        "travel_date": "travel date",
        "adults": "number of adults",
    }
    for field in ctx.missing_booking_fields:
        missing_labels.append(_FIELD_LABELS.get(field, field))

    if known_parts and missing_labels:
        detail_str = " for ".join(
            [known_parts[0]]
            + [", ".join(known_parts[1:])]
            if len(known_parts) > 1
            else known_parts
        )
        missing_str = " and ".join(missing_labels)
        return (
            f"I have your {detail_str} request noted. "
            f"To continue, please provide your {missing_str}."
        )

    if known_parts:
        detail_str = " for ".join(
            [known_parts[0]]
            + [", ".join(known_parts[1:])]
            if len(known_parts) > 1
            else known_parts
        )
        return (
            f"I have your {detail_str} request noted. "
            "Our team can review the request and confirm the next steps."
        )

    if missing_labels:
        missing_str = " and ".join(missing_labels)
        return (
            f"To continue, please provide your {missing_str}."
        )

    return get_safety_fallback(SafetyCategory.OPERATIONAL_PROMISE)


def _build_human_review_fallback(_ctx: SafetyFallbackContext) -> str:
    """Build contextual fallback for HUMAN_REVIEW stage."""
    return (
        "I have the details you've provided. "
        "This request requires review by our team before anything can be confirmed."
    )


def _build_confirmed_fallback(_ctx: SafetyFallbackContext) -> str:
    """Build contextual fallback for CONFIRMED stage."""
    return "Your booking is marked as confirmed in our system."


def _build_cancelled_fallback(_ctx: SafetyFallbackContext) -> str:
    """Build contextual fallback for CANCELLED stage."""
    return "Your booking is marked as cancelled in our system."


def build_contextual_safety_fallback(
    category: SafetyCategory,
    ctx: SafetyFallbackContext | None = None,
) -> str:
    """Build a safe fallback response, optionally using verified application state.

    Pure deterministic function: no network, no repository, no provider.

    Args:
        category: The safety category that triggered the block.
        ctx: Optional context with verified conversation state.

    Returns:
        A safe, deterministic fallback response string.
    """
    if ctx is None:
        return get_safety_fallback(category)

    # For sensitive categories, keep existing category-specific fallbacks
    # unless we can remain equally conservative with context.
    sensitive_categories = {
        SafetyCategory.PRICE,
        SafetyCategory.AVAILABILITY,
        SafetyCategory.BOOKING_CONFIRMATION,
        SafetyCategory.DISCOUNT,
        SafetyCategory.PICKUP_TIME,
        SafetyCategory.GUIDE_ASSIGNMENT,
        SafetyCategory.CONTACT_INFORMATION,
    }

    if category in sensitive_categories:
        return get_safety_fallback(category)

    # For state-aware categories, build contextual fallback
    if ctx.booking_stage == BookingStage.READY_FOR_REVIEW:
        return _build_ready_for_review_fallback(ctx)

    if ctx.booking_stage == BookingStage.COLLECTING_DETAILS:
        if ctx.requires_human:
            return _build_human_review_fallback(ctx)
        return _build_collecting_details_fallback(ctx)

    if ctx.booking_stage == BookingStage.HUMAN_REVIEW or ctx.requires_human:
        return _build_human_review_fallback(ctx)

    if ctx.booking_stage == BookingStage.CONFIRMED:
        return _build_confirmed_fallback(ctx)

    if ctx.booking_stage == BookingStage.CANCELLED:
        return _build_cancelled_fallback(ctx)

    # Default: use existing category fallback
    return get_safety_fallback(category)