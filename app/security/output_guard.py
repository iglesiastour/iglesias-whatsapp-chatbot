"""Deterministic output safety validation for AI-generated replies.

Detection only: identifies obvious unverified operational claims in an AI
reply before it would be sent to a customer. Intentionally conservative.
"""

import re
from dataclasses import dataclass

from app.models.conversation import BookingStage
from app.prompts.policies import SafetyCategory, get_safety_fallback


@dataclass(frozen=True)
class OutputSafetyResult:
    is_safe: bool
    violations: tuple[SafetyCategory, ...]


# Deterministic evaluation/reporting order for violations.
_CHECK_ORDER: tuple[SafetyCategory, ...] = (
    SafetyCategory.BOOKING_CONFIRMATION,
    SafetyCategory.AVAILABILITY,
    SafetyCategory.DISCOUNT,
    SafetyCategory.PICKUP_TIME,
    SafetyCategory.PRICE,
    SafetyCategory.CONTACT_INFORMATION,
    SafetyCategory.OPERATIONAL_PROMISE,
    SafetyCategory.UNSUPPORTED_DETAIL,
)

# Safe/negation language takes precedence over claim detection per category.
_SAFE_PHRASES: dict[SafetyCategory, tuple[str, ...]] = {
    SafetyCategory.BOOKING_CONFIRMATION: (
        "not confirmed yet",
        "not confirmed",
        "cannot confirm the booking",
        "can't confirm the booking",
        "will confirm the details",
        "to confirm it",
        "booking team will confirm",
    ),
    SafetyCategory.AVAILABILITY: (
        "checking availability",
        "cannot confirm availability",
        "can't confirm availability",
        "availability needs to be checked",
        "availability will be checked",
    ),
    SafetyCategory.DISCOUNT: (
        "check available discount options",
        "checking available discount options",
        "cannot confirm a discount",
        "can't confirm a discount",
        "discount options",
    ),
    SafetyCategory.PICKUP_TIME: (
        "check the pickup",
        "checking the pickup",
        "pickup details will be confirmed",
        "pickup details are confirmed later",
        "pickup time will be confirmed",
    ),
}

# Strong operational claims per category (case-insensitive substrings).
_CLAIM_PHRASES: dict[SafetyCategory, tuple[str, ...]] = {
    SafetyCategory.BOOKING_CONFIRMATION: (
        "your booking is confirmed",
        "your reservation is confirmed",
        "we have confirmed your booking",
        "your tour is confirmed",
        "booking is confirmed",
        "booking confirmed",
        "reservation is confirmed",
        "reservation confirmed",
    ),
    SafetyCategory.AVAILABILITY: (
        "we have availability",
        "there is availability",
        "tour is available",
        "we are available",
        "it is available",
        "yes, it is available",
        "we have seats available",
        "seats available",
    ),
    SafetyCategory.DISCOUNT: (
        "i can give you a",
        "we can offer",
        "your discount is",
        "i can apply a discount",
        "i can offer you a discount",
    ),
    SafetyCategory.OPERATIONAL_PROMISE: (
        "i'll forward this to our booking team",
        "i will forward this to our booking team",
        "forward this to our booking team",
        "forward to our booking team",
        "i'll pass this to our booking team",
        "i will pass this to our booking team",
        "pass this to our booking team",
        "i'll pass everything to our booking team",
        "i will pass everything to our booking team",
        "pass everything to our booking team",
        "our booking team will now review",
        "our team will contact you shortly",
        "they'll be in touch shortly",
        "they will be in touch shortly",
        "we'll get back to you shortly",
        "we will get back to you shortly",
        "i can check availability",
        "i'll check availability for you",
        "i will check availability for you",
        "we will check availability",
        "we'll check availability",
        "we'll send you the exact price",
        "we will send you the exact price",
        "i'll send you the price",
        "i will send you the price",
        "i'll forward this to the team",
        "i will forward this to the team",
        "i'll pass this along to our team",
        "we will review your request",
        "we'll review your request",
    ),
}

# Regex-based detection for numeric/structured claims.
_CLAIM_REGEXES: dict[SafetyCategory, tuple[re.Pattern[str], ...]] = {
    SafetyCategory.PRICE: (
        # Currency symbol attached to a number: €75, $ 90, £1,200
        re.compile(r"[€$£]\s*\d"),
        # Number followed by currency word/code: 120 USD, 80 euros
        re.compile(
            r"\d(?:[\d.,]*\d)?\s*(?:usd|eur|gbp|try|tl|euros?|dollars?|pounds?|lira)\b",
            re.IGNORECASE,
        ),
    ),
    SafetyCategory.DISCOUNT: (
        # Explicit percentage discounts: 10% discount, 15% off
        re.compile(r"\d+\s*%\s*(?:discount|off)", re.IGNORECASE),
    ),
    SafetyCategory.PICKUP_TIME: (
        re.compile(r"pickup time is \d", re.IGNORECASE),
        re.compile(r"pickup is confirmed for \d", re.IGNORECASE),
        re.compile(r"pick you up at \d", re.IGNORECASE),
        re.compile(r"will pick you up at \d", re.IGNORECASE),
        re.compile(r"driver will arrive at \d", re.IGNORECASE),
        # A confirmed booking asserting a concrete time implies a committed pickup time.
        re.compile(r"\bconfirmed\b[^.]*\bat \d{1,2}\b", re.IGNORECASE),
    ),
    SafetyCategory.CONTACT_INFORMATION: (
        # Phone numbers presented as company contact info (+international style)
        re.compile(r"(?:call us at|whatsapp(?: number)? is|contact (?:us|number))\s*:?\s*\+?\d[\d\s().-]{6,}\d", re.IGNORECASE),
        re.compile(r"\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{3}[\s.-]?\d{3,4}"),
        # Email addresses presented as company contact info
        re.compile(r"(?:email us at|our email is|e-?mail\s*:)\s*\S+@\S+\.\S+", re.IGNORECASE),
        re.compile(r"\b[\w.+-]+@(?:example|info)\.[a-z]{2,}\b|[\w.+-]+@[\w-]+\.(?:com|net|org)\b", re.IGNORECASE),
    ),
    SafetyCategory.OPERATIONAL_PROMISE: (
        # Combined action + commitment: I'll/we'll/I will/we will + action verb + this/your + target
        # These patterns match when the AI commits to taking a specific action on the customer's request.
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+"
            r"(?:forward|pass|send|contact|review)\s+"
            r"(?:this|your|the)\s+.*?"
            r"(?:booking team|our team|availability|price|pricing|request|booking)",
            re.IGNORECASE,
        ),
        # Check this/your with our booking team (safe: "check availability with" is a fallback)
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+check\s+"
            r"(?:this|your)\s+with\s+"
            r"(?:our\s+)?(?:booking team|team)",
            re.IGNORECASE,
        ),
        # Follow-up commitment patterns: get back to you shortly/soon
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+get back to you\s+(?:shortly|soon|asap)",
            re.IGNORECASE,
        ),
        # Contact you shortly/soon
        re.compile(
            r"(?:will\s+)?contact you\s+(?:shortly|soon|asap)",
            re.IGNORECASE,
        ),
        # Be in touch shortly/soon
        re.compile(
            r"(?:will\s+)?be in touch\s+(?:shortly|soon|asap)",
            re.IGNORECASE,
        ),
        # Check availability/pricing/price with team/booking team
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+check\s+"
            r"(?:the\s+)?(?:availability|pricing|price)\s*"
            r"(?:and\s+(?:the\s+)?(?:availability|pricing|price)\s*)?"
            r"(?:with\s+(?:our\s+)?(?:booking\s+team|team))",
            re.IGNORECASE,
        ),
        # Check availability/pricing/price and let you know/get back to you
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+check\s+"
            r"(?:the\s+)?(?:availability|pricing|price)\s*"
            r"(?:and\s+(?:let you know|get back to you|contact you))",
            re.IGNORECASE,
        ),
        # Check availability with team and let you know
        re.compile(
            r"(?:i(?:'|'ll| will)|we(?:'|'ll| will))\s+check\s+"
            r"(?:the\s+)?availability\s+"
            r"(?:with\s+(?:our\s+)?(?:booking\s+team|team))?\s*"
            r"(?:and\s+(?:let you know|get back to you|contact you))",
            re.IGNORECASE,
        ),
    ),
}


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, normalize curly apostrophes.

    Handles I'll / I\u2019ll / We\u2019ll / they\u2019ll consistently
    so safety matching works regardless of quotation-mark encoding.
    """
    normalized = text.casefold()
    # Normalize curly/typographic apostrophes to straight apostrophe
    normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
    normalized = normalized.replace("\u2032", "'").replace("\u2033", "'")
    return " ".join(normalized.split())


def _check_unsupported_tour_qualifier(
    text: str, known_tour: str | None
) -> bool:
    """Check if text adds unsupported qualifiers to the known tour.

    Only triggers when known_tour is provided and the reply mentions the
    tour with qualifiers not present in the known tour value.

    Safe patterns (always allowed):
    - "Ephesus tour" (exact)
    - "the Ephesus tour"
    - "your Ephesus tour"
    - "an Ephesus tour"

    Unsafe patterns (flagged):
    - "private Ephesus tour"
    - "luxury Ephesus tour"
    - "VIP Ephesus tour"
    """
    if not known_tour:
        return False

    normalized_text = _normalize(text)
    normalized_tour = _normalize(known_tour)

    # If the normalized tour doesn't appear in the text, no qualifier check needed.
    if normalized_tour not in normalized_text:
        return False

    # Safe determiners that can precede the tour without adding new meaning
    _SAFE_DETERMINERS = ("the", "your", "a", "an")

    # Words that are part of the tour name itself
    tour_words = set(normalized_tour.split())

    # Find all positions where the tour appears in the text
    tour_start = 0
    while True:
        pos = normalized_text.find(normalized_tour, tour_start)
        if pos == -1:
            break

        # Check what comes immediately before the tour
        prefix = normalized_text[:pos].strip()
        if prefix:
            # Get the last word(s) before the tour
            prefix_words = prefix.split()
            last_prefix_word = prefix_words[-1] if prefix_words else ""

            # If the last prefix word is a safe determiner, it's OK
            if last_prefix_word in _SAFE_DETERMINERS:
                tour_start = pos + len(normalized_tour)
                continue

            # If the last prefix word is part of the tour name, check the word before that
            if last_prefix_word in tour_words and len(prefix_words) > 1:
                second_last = prefix_words[-2]
                if second_last in _SAFE_DETERMINERS:
                    tour_start = pos + len(normalized_tour)
                    continue

            # If we get here, there's a qualifier before the tour
            return True

        tour_start = pos + len(normalized_tour)

    return False


def _check_optional_field_question(
    text: str,
    booking_stage: BookingStage | None,
) -> bool:
    """Check if text asks about optional fields when in READY_FOR_REVIEW stage.

    Only active when booking_stage is READY_FOR_REVIEW.
    Detects questions about children, hotel, cruise ship, pickup location,
    preferred language.
    """
    if booking_stage is not BookingStage.READY_FOR_REVIEW:
        return False

    normalized = _normalize(text)

    # Patterns for optional field questions
    optional_field_patterns = [
        # Children
        re.compile(r"(?:how many|number of)\s+children", re.IGNORECASE),
        re.compile(r"children\s+(?:will|are|joining|coming|attending)", re.IGNORECASE),
        re.compile(r"any\s+children", re.IGNORECASE),
        re.compile(r"bringing\s+children", re.IGNORECASE),
        # Hotel
        re.compile(r"(?:which|what)\s+hotel", re.IGNORECASE),
        re.compile(r"hotel\s+(?:are|is|will|staying)", re.IGNORECASE),
        re.compile(r"staying\s+(?:at|in)\s+(?:which|what)\s+hotel", re.IGNORECASE),
        re.compile(r"hotel\s+name", re.IGNORECASE),
        # Cruise ship
        re.compile(r"(?:which|what)\s+cruise\s+ship", re.IGNORECASE),
        re.compile(r"cruise\s+ship\s+(?:are|is|will)", re.IGNORECASE),
        re.compile(r"arriving\s+(?:by|on)\s+cruise", re.IGNORECASE),
        re.compile(r"cruise\s+ship\s+name", re.IGNORECASE),
        # Pickup location
        re.compile(r"(?:which|what)\s+pickup\s+location", re.IGNORECASE),
        re.compile(r"pickup\s+location\s+(?:are|is|will)", re.IGNORECASE),
        re.compile(r"where\s+(?:are|is)\s+you\s+(?:staying|located)", re.IGNORECASE),
        re.compile(r"pickup\s+address", re.IGNORECASE),
        # Preferred language
        re.compile(r"(?:which|what)\s+language", re.IGNORECASE),
        re.compile(r"preferred\s+language", re.IGNORECASE),
        re.compile(r"language\s+(?:do|does|would|prefer)", re.IGNORECASE),
        re.compile(r"speak\s+(?:which|what)\s+language", re.IGNORECASE),
    ]

    # Check if any pattern matches
    for pattern in optional_field_patterns:
        if pattern.search(normalized):
            return True

    return False


def inspect_ai_output(
    text: str,
    known_tour: str | None = None,
    booking_stage: BookingStage | None = None,
) -> OutputSafetyResult:
    """Detect obvious unverified operational claims in an AI reply."""
    normalized = _normalize(text)

    violations: list[SafetyCategory] = []

    for category in _CHECK_ORDER:
        # Our own approved fallback responses are always considered safe.
        if normalized == _normalize(get_safety_fallback(category)):
            continue

        if any(phrase in normalized for phrase in _SAFE_PHRASES.get(category, ())):
            continue

        claimed = any(phrase in normalized for phrase in _CLAIM_PHRASES.get(category, ()))
        if not claimed:
            claimed = any(regex.search(normalized) for regex in _CLAIM_REGEXES.get(category, ()))

        if claimed and category not in violations:
            violations.append(category)

    # Check for unsupported tour qualifier (only when known_tour is provided)
    if known_tour and _check_unsupported_tour_qualifier(text, known_tour):
        if SafetyCategory.UNSUPPORTED_DETAIL not in violations:
            violations.append(SafetyCategory.UNSUPPORTED_DETAIL)

    # Check for optional field reask in READY_FOR_REVIEW stage
    if _check_optional_field_question(text, booking_stage):
        if SafetyCategory.OPTIONAL_FIELD_REASK not in violations:
            violations.append(SafetyCategory.OPTIONAL_FIELD_REASK)

    return OutputSafetyResult(
        is_safe=not violations,
        violations=tuple(violations),
    )
