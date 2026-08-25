"""Typed domain models for conversation intent and state.

Deterministic data-model foundation only: no intent detection, persistence,
or route integration in this step.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, Field


class ConversationIntent(StrEnum):
    GREETING = "greeting"
    GENERAL_QUESTION = "general_question"
    TOUR_INFORMATION = "tour_information"
    PRICE_REQUEST = "price_request"
    AVAILABILITY_REQUEST = "availability_request"
    BOOKING_REQUEST = "booking_request"
    EXISTING_BOOKING = "existing_booking"
    CANCELLATION_REQUEST = "cancellation_request"
    COMPLAINT = "complaint"
    HUMAN_REQUEST = "human_request"
    UNSUPPORTED = "unsupported"


class BookingStage(StrEnum):
    NONE = "none"
    COLLECTING_DETAILS = "collecting_details"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_REVIEW = "human_review"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


# Required fields for a booking request, in canonical order.
_REQUIRED_BOOKING_FIELDS: tuple[str, ...] = ("tour", "travel_date", "adults")

# Intents that always require human involvement.
_HUMAN_REQUIRED_INTENTS: frozenset[ConversationIntent] = frozenset(
    {
        ConversationIntent.COMPLAINT,
        ConversationIntent.HUMAN_REQUEST,
        ConversationIntent.CANCELLATION_REQUEST,
    }
)


class ConversationState(BaseModel):
    intent: ConversationIntent = ConversationIntent.GENERAL_QUESTION

    tour: str | None = None
    travel_date: date | None = None

    adults: int | None = Field(default=None, ge=1, le=100)
    children: int | None = Field(default=None, ge=0, le=100)

    cruise_ship: str | None = None
    hotel: str | None = None
    pickup_location: str | None = None

    preferred_language: str | None = None

    booking_stage: BookingStage = BookingStage.NONE
    needs_human: bool = False

    def missing_booking_fields(self) -> tuple[str, ...]:
        """Return required booking field names that are not set, in canonical order."""
        return tuple(
            field for field in _REQUIRED_BOOKING_FIELDS if getattr(self, field) is None
        )

    @property
    def is_booking_ready(self) -> bool:
        """True only when this is a booking request with all required fields set."""
        return (
            self.intent is ConversationIntent.BOOKING_REQUEST
            and self.missing_booking_fields() == ()
        )

    @property
    def requires_human(self) -> bool:
        """True when human handoff is needed (explicit flag or specific intents)."""
        if self.needs_human:
            return True
        return self.intent in _HUMAN_REQUIRED_INTENTS
