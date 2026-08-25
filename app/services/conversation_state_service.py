"""Deterministic conversation state transition service.

Updates ConversationState from a customer message using the deterministic
intent classifier. Pure state transition only: no AI provider calls, no
entity extraction, no persistence. The incoming state is never mutated.
"""

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.services.intent_classifier import classify_intent

_AUTHORITATIVE_STAGES: frozenset[BookingStage] = frozenset(
    {BookingStage.CONFIRMED, BookingStage.CANCELLED}
)
_HUMAN_REQUIRED_INTENTS: frozenset[ConversationIntent] = frozenset(
    {
        ConversationIntent.COMPLAINT,
        ConversationIntent.HUMAN_REQUEST,
        ConversationIntent.CANCELLATION_REQUEST,
    }
)
_BOOKING_ACTIVE_STAGES: frozenset[BookingStage] = frozenset(
    {BookingStage.COLLECTING_DETAILS, BookingStage.READY_FOR_REVIEW}
)
_BOOKING_PRESERVING_INTENTS: frozenset[ConversationIntent] = frozenset(
    {
        ConversationIntent.PRICE_REQUEST,
        ConversationIntent.AVAILABILITY_REQUEST,
    }
)


def apply_message_to_state(
    state: ConversationState,
    message: str,
) -> ConversationState:
    """Return a NEW ConversationState updated from the customer message.

    The incoming state object is never mutated.
    """
    intent = classify_intent(message)

    # Human escalation is one-way inside this service.
    needs_human = state.needs_human or intent in _HUMAN_REQUIRED_INTENTS

    # Intermediate merge so readiness checks see the new intent together
    # with the preserved entity fields (no extraction happens here).
    merged = state.model_copy(update={"intent": intent, "needs_human": needs_human})

    booking_stage = _next_booking_stage(state.booking_stage, intent, merged)

    return state.model_copy(
        update={
            "intent": intent,
            "needs_human": needs_human,
            "booking_stage": booking_stage,
        },
    )


def _next_booking_stage(
    current: BookingStage,
    intent: ConversationIntent,
    merged: ConversationState,
) -> BookingStage:
    # Authoritative business states are never changed by customer messages.
    if current in _AUTHORITATIVE_STAGES:
        return current

    if intent is ConversationIntent.BOOKING_REQUEST:
        if merged.missing_booking_fields():
            return BookingStage.COLLECTING_DETAILS
        return BookingStage.READY_FOR_REVIEW

    if intent in _HUMAN_REQUIRED_INTENTS and current in _BOOKING_ACTIVE_STAGES:
        return BookingStage.HUMAN_REVIEW

    if (
        current is BookingStage.COLLECTING_DETAILS
        and intent in _BOOKING_PRESERVING_INTENTS
    ):
        return BookingStage.COLLECTING_DETAILS

    # EXISTING_BOOKING, GREETING, GENERAL_QUESTION, TOUR_INFORMATION,
    # PRICE_REQUEST / AVAILABILITY_REQUEST outside an active booking, etc.
    # all preserve the current stage.
    return current
