"""Deterministic output safety validation for AI-generated replies.

Detection only: identifies obvious unverified operational claims in an AI
reply before it would be sent to a customer. Intentionally conservative.
"""

import re
from dataclasses import dataclass

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
        "check availability",
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
}


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace without mutating the original."""
    return " ".join(text.casefold().split())


def inspect_ai_output(text: str) -> OutputSafetyResult:
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

    return OutputSafetyResult(
        is_safe=not violations,
        violations=tuple(violations),
    )
