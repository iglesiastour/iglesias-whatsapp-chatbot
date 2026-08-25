"""Deterministic rule-based intent classifier for obvious customer intents.

Conservative by design: ambiguous or unmatched messages fall back to
GENERAL_QUESTION rather than aggressive classification.
"""

from app.models.conversation import ConversationIntent

# Checked in explicit priority order; first match wins.
_PHRASES_BY_PRIORITY: tuple[tuple[ConversationIntent, tuple[str, ...]], ...] = (
    (
        ConversationIntent.HUMAN_REQUEST,
        (
            "talk to a human",
            "speak to a human",
            "speak with a human",
            "talk to a real person",
            "speak to a real person",
            "human agent",
            "talk to an agent",
            "speak to an agent",
            "customer service",
            "representative",
            "someone from your team",
            "your booking team",
        ),
    ),
    (
        ConversationIntent.COMPLAINT,
        (
            "want to complain",
            "have a complaint",
            "this is unacceptable",
            "very unhappy",
            "disappointed with the service",
            "terrible service",
            "bad experience",
        ),
    ),
    (
        ConversationIntent.CANCELLATION_REQUEST,
        (
            "cancel my booking",
            "cancel my reservation",
            "cancel our booking",
            "cancel our reservation",
            "cancel the tour",
            "want to cancel",
            "please cancel",
        ),
    ),
    (
        ConversationIntent.EXISTING_BOOKING,
        (
            "existing booking",
            "my booking",
            "my reservation",
            "our booking",
            "our reservation",
            "booking reference",
            "reservation number",
            "booking number",
            "already booked",
            "have a booking",
            "have a reservation",
        ),
    ),
    (
        ConversationIntent.BOOKING_REQUEST,
        (
            "want to book",
            "would like to book",
            "like to book",
            "can i book",
            "can we book",
            "book a tour",
            "book this tour",
            "book the tour",
            "reserve a tour",
            "reserve the tour",
            "make a booking",
            "make a reservation",
        ),
    ),
    (
        ConversationIntent.PRICE_REQUEST,
        (
            "how much",
            "what is the price",
            "what's the price",
            "what does it cost",
            "price",
            "pricing",
            "cost",
        ),
    ),
    (
        ConversationIntent.AVAILABILITY_REQUEST,
        (
            "is it available",
            "are you available",
            "do you have availability",
            "any availability",
            "availability",
            "available tomorrow",
            "do you have space",
            "any seats available",
            "seats available",
            "available on",
            "available for",
        ),
    ),
    (
        ConversationIntent.TOUR_INFORMATION,
        (
            "ephesus",
            "pamukkale",
            "cappadocia",
            "istanbul",
            "biblical tour",
            "shore excursion",
            "airport transfer",
            "turkey package",
            "tour",
            "guide",
        ),
    ),
)

_GREETINGS: frozenset[str] = frozenset(
    {
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "merhaba",
        "selam",
        "hi there",
        "hello there",
        "hey there",
    }
)


def _normalize(message: str) -> str:
    """Casefold and collapse whitespace without mutating the original."""
    return " ".join(message.casefold().split())


def classify_intent(message: str) -> ConversationIntent:
    """Classify an obvious customer intent deterministically.

    Ambiguous or unmatched messages return GENERAL_QUESTION.
    """
    normalized = _normalize(message)
    if not normalized:
        return ConversationIntent.GENERAL_QUESTION

    for intent, phrases in _PHRASES_BY_PRIORITY:
        if any(phrase in normalized for phrase in phrases):
            return intent

    # Short standalone greetings only — never overrides a substantive request
    # (substantive intents are already matched above).
    stripped = normalized.strip("!.?, ")
    if stripped in _GREETINGS:
        return ConversationIntent.GREETING

    return ConversationIntent.GENERAL_QUESTION
