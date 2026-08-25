"""Deterministic tests for the rule-based intent classifier."""

import os

import pytest

from app.models.conversation import ConversationIntent
from app.services.intent_classifier import classify_intent


def test_empty_message_returns_general_question() -> None:
    assert classify_intent("") is ConversationIntent.GENERAL_QUESTION


def test_whitespace_only_returns_general_question() -> None:
    assert classify_intent("   ") is ConversationIntent.GENERAL_QUESTION


@pytest.mark.parametrize(
    "message",
    ["hi", "hello", "hey", "good morning", "good evening", "merhaba", "selam"],
)
def test_greeting_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.GREETING


def test_greeting_with_price_request_is_price_request() -> None:
    assert (
        classify_intent("Hello, how much is the Ephesus tour?")
        is ConversationIntent.PRICE_REQUEST
    )


def test_simple_greeting_with_extra_word_is_greeting() -> None:
    assert classify_intent("Hi there!") is ConversationIntent.GREETING
    assert classify_intent("Hello!!") is ConversationIntent.GREETING


@pytest.mark.parametrize(
    "message",
    [
        "I want to talk to a human.",
        "Can I speak to a real person?",
        "Please connect me with customer service.",
        "Can someone from your team help me?",
        "Let me talk to an agent.",
    ],
)
def test_human_request_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.HUMAN_REQUEST


def test_guide_real_person_mention_is_not_human_request() -> None:
    assert (
        classify_intent("Is the guide a real person?")
        is not ConversationIntent.HUMAN_REQUEST
    )
    assert (
        classify_intent("Is the guide a real person?")
        is ConversationIntent.TOUR_INFORMATION
    )


@pytest.mark.parametrize(
    "message",
    [
        "I want to complain about the service.",
        "I have a complaint regarding my tour.",
        "This is unacceptable.",
        "Terrible service, I am very unhappy.",
        "Bad experience overall.",
    ],
)
def test_complaint_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.COMPLAINT


@pytest.mark.parametrize(
    "message",
    [
        "Cancel my booking please.",
        "I want to cancel my reservation.",
        "Please cancel the tour.",
        "We need to cancel our booking.",
    ],
)
def test_cancellation_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.CANCELLATION_REQUEST


@pytest.mark.parametrize(
    "message",
    [
        "I have a booking for tomorrow.",
        "What time is my reservation?",
        "Here is my booking reference: ABC123.",
        "We already booked with you.",
        "I have a reservation question.",
    ],
)
def test_existing_booking_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.EXISTING_BOOKING


def test_cancellation_outranks_existing_booking() -> None:
    assert (
        classify_intent("Cancel my booking from last week")
        is ConversationIntent.CANCELLATION_REQUEST
    )


@pytest.mark.parametrize(
    "message",
    [
        "I want to book a tour.",
        "We would like to book the Ephesus tour.",
        "Can I book for tomorrow?",
        "Reserve a tour for four people.",
        "Make a reservation please.",
    ],
)
def test_booking_request_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.BOOKING_REQUEST


def test_how_booking_works_is_not_booking_request() -> None:
    assert (
        classify_intent("How does booking work?")
        is not ConversationIntent.BOOKING_REQUEST
    )


def test_booking_outranks_availability() -> None:
    assert (
        classify_intent("Can I book if it is available tomorrow?")
        is ConversationIntent.BOOKING_REQUEST
    )


@pytest.mark.parametrize(
    "message",
    [
        "How much is the Ephesus tour?",
        "What is the price for two people?",
        "What does it cost?",
        "Do you have pricing information?",
    ],
)
def test_price_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.PRICE_REQUEST


@pytest.mark.parametrize(
    "message",
    [
        "Is it available on Monday?",
        "Do you have availability next week?",
        "Are you available tomorrow?",
        "Any seats available for the Pamukkale tour?",
    ],
)
def test_availability_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.AVAILABILITY_REQUEST


@pytest.mark.parametrize(
    "message",
    [
        "Tell me about Ephesus.",
        "What is included in the Pamukkale tour?",
        "Do you offer Cappadocia tours?",
        "Tell us about the shore excursion options.",
    ],
)
def test_tour_information_examples(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.TOUR_INFORMATION


@pytest.mark.parametrize(
    "message",
    [
        "What are your office hours?",
        "What should I bring with me?",
        "How long has your company been operating?",
    ],
)
def test_unknown_normal_question_is_general_question(message: str) -> None:
    assert classify_intent(message) is ConversationIntent.GENERAL_QUESTION


def test_case_insensitive() -> None:
    assert classify_intent("HOW MUCH IS THE TOUR?") is ConversationIntent.PRICE_REQUEST
    assert (
        classify_intent("CANCEL MY BOOKING") is ConversationIntent.CANCELLATION_REQUEST
    )


def test_whitespace_normalization() -> None:
    assert (
        classify_intent("  how   much \t is it?  ") is ConversationIntent.PRICE_REQUEST
    )
    assert classify_intent("\nhi\n") is ConversationIntent.GREETING


def test_original_message_not_mutated() -> None:
    original = "  HOW MUCH Is The Tour?  "
    classify_intent(original)
    assert original == "  HOW MUCH Is The Tour?  "


@pytest.mark.parametrize(
    "message",
    ["", "   ", "hello", "how much", "cancel my booking", "tell me about ephesus"],
)
def test_classifier_returns_conversation_intent_values(message: str) -> None:
    assert isinstance(classify_intent(message), ConversationIntent)


def test_deterministic_repeated_calls() -> None:
    message = "I want to book a tour"
    results = {classify_intent(message) for _ in range(10)}
    assert results == {ConversationIntent.BOOKING_REQUEST}


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    classify_intent("hello")
    assert dict(os.environ) == snapshot
