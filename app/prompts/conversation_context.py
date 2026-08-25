"""Deterministic builder converting ConversationState into reply context.

Produces a safe, concise prompt context for customer-facing reply generation,
using only business-relevant facts already known in state. No internal state,
ids, timestamps, or raw JSON are exposed.
"""

import re

from app.models.conversation import BookingStage, ConversationState

_FIELD_ORDER: tuple[tuple[str, str], ...] = (
    ("intent", "Current intent"),
    ("tour", "Tour"),
    ("travel_date", "Travel date"),
    ("adults", "Adults"),
    ("children", "Children"),
    ("cruise_ship", "Cruise ship"),
    ("hotel", "Hotel"),
    ("pickup_location", "Pickup location"),
    ("preferred_language", "Preferred language"),
)

_MISSING_LABELS: dict[str, str] = {
    "tour": "tour",
    "travel_date": "travel date",
    "adults": "adults",
}

_SAFETY_REMINDER = (
    "Use only the known details above. Do not invent price, availability, "
    "contact details, booking confirmation, pickup time, guide assignment, "
    "vehicle assignment, discount, or payment status.\n"
    "Do not claim that a booking-team action, callback, price quote, "
    "availability check, or follow-up has already been initiated unless "
    "the application state explicitly says so.\n"
    "Do not add qualifiers or product details to the known tour that are "
    "not present in the known details."
)

_STAGE_LABEL: dict[BookingStage, str] = {
    BookingStage.NONE: "none",
    BookingStage.COLLECTING_DETAILS: "collecting_details",
    BookingStage.READY_FOR_REVIEW: "ready_for_review",
    BookingStage.HUMAN_REVIEW: "human_review",
    BookingStage.CONFIRMED: "confirmed",
    BookingStage.CANCELLED: "cancelled",
}

_KNOWN_NAME_INSTRUCTION = (
    "The customer name above is application-provided and may be used "
    "naturally when appropriate. Do not use any other customer name."
)

_UNKNOWN_NAME_INSTRUCTION = (
    "Customer name is not known. "
    "Do not invent, guess, infer, or use a customer name."
)


def _format_date(value) -> str:
    return value.isoformat()


def _normalize_customer_name(name: str | None) -> str | None:
    """Normalize customer name conservatively.

    Strip leading/trailing whitespace, collapse internal whitespace,
    preserve original capitalization. Empty/whitespace-only → None.
    """
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    if not stripped:
        return None
    return re.sub(r"\s+", " ", stripped)


def build_conversation_context(
    state: ConversationState,
    customer_name: str | None = None,
) -> str:
    normalized_name = _normalize_customer_name(customer_name)

    lines: list[str] = ["Known conversation details:"]

    known: list[str] = []

    if normalized_name is not None:
        known.append(f"- Customer name: {normalized_name}")

    for field, label in _FIELD_ORDER:
        value = getattr(state, field)
        if value is None:
            continue
        display = _format_date(value) if field == "travel_date" else str(value)
        known.append(f"- {label}: {display}")

    if state.booking_stage is not BookingStage.NONE:
        known.append(f"- Booking stage: {_STAGE_LABEL[state.booking_stage]}")

    if state.needs_human:
        known.append("- Needs human follow-up: yes")

    if known:
        lines.extend(known)
    else:
        lines.append("- No booking details are known yet.")

    lines.append("")
    lines.append("Use only verified information from the conversation.")

    if normalized_name is not None:
        lines.append(_KNOWN_NAME_INSTRUCTION)
    else:
        lines.append(_UNKNOWN_NAME_INSTRUCTION)

    stage = state.booking_stage
    if stage is BookingStage.READY_FOR_REVIEW:
        lines = _append_ready_for_review(lines)
    elif stage is BookingStage.COLLECTING_DETAILS:
        lines = _append_collecting_details(lines, state)
    elif stage is BookingStage.HUMAN_REVIEW or state.requires_human:
        lines = _append_human_review(lines)
    elif stage is BookingStage.CONFIRMED:
        lines.append("Backend status: confirmed. Do not change or reinterpret this status.")
    elif stage is BookingStage.CANCELLED:
        lines.append("Backend status: cancelled. Do not change or reinterpret this status.")

    lines.append("")
    lines.append(_SAFETY_REMINDER)

    return "\n".join(lines)


def _append_ready_for_review(lines: list[str]) -> list[str]:
    lines.append(
        "The required booking details already known are complete. "
        "Do not ask again for tour, travel date, or adults if they are present. "
        "Do not ask for additional optional details such as children, cruise ship, "
        "hotel, pickup location, or preferred language. "
        "Guide the customer toward review/human confirmation instead."
    )
    return lines


def _append_collecting_details(
    lines: list[str],
    state: ConversationState,
) -> list[str]:
    missing = state.missing_booking_fields()
    if missing:
        lines.append("Still needed for booking:")
        for field in missing:
            lines.append(f"- {_MISSING_LABELS.get(field, field)}")
    lines.append(
        "Ask only for missing required booking information. "
        "Do not ask again for booking details already known."
    )
    return lines


def _append_human_review(lines: list[str]) -> list[str]:
    lines.append(
        "This conversation requires human follow-up. "
        "Do not make new operational commitments. "
        "Keep the reply concise and route the customer toward the team."
    )
    return lines