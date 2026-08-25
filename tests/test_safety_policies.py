"""Deterministic tests for the safety policy primitives (no network, no env)."""

import pytest

from app.prompts.policies import (
    FORBIDDEN_UNVERIFIED_FACTS,
    SAFETY_FALLBACKS,
    SafetyCategory,
    get_safety_fallback,
)

MANDATORY_FORBIDDEN_FACTS = (
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


def test_every_safety_category_has_a_fallback() -> None:
    assert set(SAFETY_FALLBACKS.keys()) == set(SafetyCategory)


def test_every_fallback_is_a_non_empty_string() -> None:
    for category in SafetyCategory:
        fallback = SAFETY_FALLBACKS[category]
        assert isinstance(fallback, str)
        assert fallback.strip() != ""


@pytest.mark.parametrize("category", list(SafetyCategory))
def test_get_safety_fallback_returns_expected_value(category: SafetyCategory) -> None:
    assert get_safety_fallback(category) == SAFETY_FALLBACKS[category]


def test_forbidden_unverified_facts_contain_mandatory_categories() -> None:
    for fact in MANDATORY_FORBIDDEN_FACTS:
        assert fact in FORBIDDEN_UNVERIFIED_FACTS


def test_no_duplicate_forbidden_fact_entries() -> None:
    assert len(FORBIDDEN_UNVERIFIED_FACTS) == len(set(FORBIDDEN_UNVERIFIED_FACTS))


def test_unknown_category_raises_key_error() -> None:
    with pytest.raises(KeyError):
        get_safety_fallback("not-a-real-category")  # type: ignore[arg-value]
