"""Deterministic tests for the conversation pipeline service."""

import asyncio
import os
from datetime import date

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.extraction import ExtractedEntities, StructuredExtraction
from app.services.ai.base import AIProvider, AIProviderError
from app.services.conversation_pipeline_service import ConversationPipelineService


class PipelineFakeProvider(AIProvider):
    """Records calls; returns configured extraction or raises configured error."""

    def __init__(
        self,
        entities: ExtractedEntities | None = None,
        exception: Exception | None = None,
    ):
        self.entities = entities if entities is not None else ExtractedEntities()
        self.exception = exception
        self.extract_calls: list[str] = []
        self.reply_calls: list[str] = []

    async def generate_reply(self, message: str) -> str:
        self.reply_calls.append(message)
        return "should not be called"

    async def extract_entities(self, message: str) -> StructuredExtraction:
        self.extract_calls.append(message)
        if self.exception is not None:
            raise self.exception
        return StructuredExtraction(entities=self.entities)


def run(service: ConversationPipelineService, state: ConversationState, message: str):
    return asyncio.run(service.process_message(state, message))


# --- Immutability / identity ---


def test_incoming_state_not_mutated() -> None:
    state = ConversationState(tour="Ephesus")
    snapshot = state.model_copy()
    run(ConversationPipelineService(PipelineFakeProvider()), state, "hello")
    assert state == snapshot


def test_new_state_returned_and_valid() -> None:
    provider = PipelineFakeProvider()
    state = ConversationState()
    result = run(ConversationPipelineService(provider), state, "hello")
    assert result is not state
    assert isinstance(result, ConversationState)


# --- Booking flow ---


def test_safe_booking_flow_runs_extraction() -> None:
    provider = PipelineFakeProvider()
    run(ConversationPipelineService(provider), ConversationState(), "I want to book")
    assert len(provider.extract_calls) == 1


def test_provider_gets_original_message() -> None:
    provider = PipelineFakeProvider()
    message = "I want to book for September 10 for 2 adults."
    run(ConversationPipelineService(provider), ConversationState(), message)
    assert provider.extract_calls == [message]


def test_booking_extraction_completes_required_fields() -> None:
    provider = PipelineFakeProvider(
        entities=ExtractedEntities(travel_date="2026-09-10", adults=2)
    )
    state = ConversationState(
        intent=ConversationIntent.GENERAL_QUESTION,
        tour="Ephesus",
        booking_stage=BookingStage.COLLECTING_DETAILS,
    )
    result = run(ConversationPipelineService(provider), state, "I want to book.")
    assert result.intent is ConversationIntent.BOOKING_REQUEST
    assert result.travel_date == date(2026, 9, 10)
    assert result.adults == 2
    assert result.booking_stage is BookingStage.READY_FOR_REVIEW


def test_partial_extraction_remains_collecting_details() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(adults=2))
    state = ConversationState(booking_stage=BookingStage.NONE)
    result = run(ConversationPipelineService(provider), state, "I want to book")
    assert result.booking_stage is BookingStage.COLLECTING_DETAILS


# --- Merge semantics through the pipeline ---


def test_existing_entity_preserved_when_extraction_none() -> None:
    provider = PipelineFakeProvider()  # all-None entities
    state = ConversationState(hotel="Korumar Hotel")
    result = run(ConversationPipelineService(provider), state, "How much?")
    assert result.hotel == "Korumar Hotel"


def test_extracted_non_none_value_overwrites_existing() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(hotel="New Hotel"))
    state = ConversationState(hotel="Old Hotel")
    result = run(ConversationPipelineService(provider), state, "Tell me about Ephesus.")
    assert result.hotel == "New Hotel"


def test_adults_correction_works() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(adults=4))
    state = ConversationState(adults=2)
    result = run(ConversationPipelineService(provider), state, "We are actually 4 people.")
    assert result.adults == 4


def test_travel_date_correction_works() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(travel_date="2026-11-01"))
    state = ConversationState(travel_date=date(2026, 9, 10))
    result = run(ConversationPipelineService(provider), state, "Actually November first.")
    assert result.travel_date == date(2026, 11, 1)


# --- Extraction skip rules ---


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("hello", ConversationIntent.GREETING),
        ("I want to talk to a human.", ConversationIntent.HUMAN_REQUEST),
        ("This is unacceptable.", ConversationIntent.COMPLAINT),
        ("Cancel my booking", ConversationIntent.CANCELLATION_REQUEST),
    ],
)
def test_skipped_intents_do_not_call_extraction(message: str, intent: ConversationIntent):
    provider = PipelineFakeProvider()
    result = run(ConversationPipelineService(provider), ConversationState(), message)
    assert result.intent is intent
    assert provider.extract_calls == []


# --- Human escalation ---


def test_human_request_sets_needs_human_and_human_review() -> None:
    provider = PipelineFakeProvider()
    state = ConversationState(booking_stage=BookingStage.COLLECTING_DETAILS)
    result = run(
        ConversationPipelineService(provider), state, "I want to speak to a human."
    )
    assert result.intent is ConversationIntent.HUMAN_REQUEST
    assert result.needs_human is True
    assert result.booking_stage is BookingStage.HUMAN_REVIEW
    assert provider.extract_calls == []


def test_complaint_sets_needs_human() -> None:
    provider = PipelineFakeProvider()
    result = run(
        ConversationPipelineService(provider),
        ConversationState(),
        "This is unacceptable.",
    )
    assert result.needs_human is True


def test_cancellation_sets_needs_human() -> None:
    provider = PipelineFakeProvider()
    result = run(
        ConversationPipelineService(provider), ConversationState(), "Cancel my booking"
    )
    assert result.needs_human is True



# --- Extraction runs for allowed intents ---


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("What are your office hours?", ConversationIntent.GENERAL_QUESTION),
        ("Tell me about Ephesus.", ConversationIntent.TOUR_INFORMATION),
        ("How much is the tour?", ConversationIntent.PRICE_REQUEST),
        ("Is it available on Monday?", ConversationIntent.AVAILABILITY_REQUEST),
        ("I want to book a tour.", ConversationIntent.BOOKING_REQUEST),
        ("I have a booking for tomorrow.", ConversationIntent.EXISTING_BOOKING),
    ],
)
def test_allowed_intents_run_extraction(message: str, intent: ConversationIntent):
    provider = PipelineFakeProvider()
    result = run(ConversationPipelineService(provider), ConversationState(), message)
    assert result.intent is intent
    assert len(provider.extract_calls) == 1


# --- Provider errors ---


def test_provider_extraction_failure_propagates() -> None:
    provider = PipelineFakeProvider(exception=AIProviderError("provider failed"))
    with pytest.raises(AIProviderError, match="provider failed"):
        run(ConversationPipelineService(provider), ConversationState(), "How much?")


def test_no_partial_result_on_provider_failure() -> None:
    provider = PipelineFakeProvider(exception=AIProviderError("provider failed"))
    state = ConversationState(hotel="Korumar Hotel")
    service = ConversationPipelineService(provider)
    try:
        asyncio.run(service.process_message(state, "How much?"))
    except AIProviderError:
        pass
    # Original state untouched; nothing persisted anywhere.
    assert state.hotel == "Korumar Hotel"


def test_service_never_calls_generate_reply_or_get_ai_provider() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(tour="Ephesus"))
    run(
        ConversationPipelineService(provider),
        ConversationState(),
        "Tell me about Ephesus.",
    )
    assert provider.reply_calls == []


# --- Authoritative stages ---


@pytest.mark.parametrize("stage", [BookingStage.CONFIRMED, BookingStage.CANCELLED])
def test_authoritative_stages_remain_unchanged(stage: BookingStage) -> None:
    provider = PipelineFakeProvider(
        entities=ExtractedEntities(travel_date="2026-10-01")
    )
    state = ConversationState(booking_stage=stage)
    result = run(ConversationPipelineService(provider), state, "I want to book")
    assert result.booking_stage is stage
    assert result.travel_date == date(2026, 10, 1)


def test_human_review_remains_human_review_through_pipeline() -> None:
    provider = PipelineFakeProvider(entities=ExtractedEntities(children=2))
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        booking_stage=BookingStage.HUMAN_REVIEW,
    )
    result = run(ConversationPipelineService(provider), state, "We have 2 children.")
    assert result.booking_stage is BookingStage.HUMAN_REVIEW
    assert result.children == 2


# --- Misc ---


def test_empty_extraction_preserves_all_entities() -> None:
    provider = PipelineFakeProvider()
    state = ConversationState(
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Ship",
        hotel="Hotel",
        pickup_location="Port",
        preferred_language="English",
    )
    snapshot = state.model_copy()
    result = run(ConversationPipelineService(provider), state, "How much?")
    # Intent may update; all entity fields must be preserved.
    assert (
        result.tour,
        result.travel_date,
        result.adults,
        result.children,
        result.cruise_ship,
        result.hotel,
        result.pickup_location,
        result.preferred_language,
    ) == (
        snapshot.tour,
        snapshot.travel_date,
        snapshot.adults,
        snapshot.children,
        snapshot.cruise_ship,
        snapshot.hotel,
        snapshot.pickup_location,
        snapshot.preferred_language,
    )


def test_all_eight_entities_flow_through_pipeline() -> None:
    provider = PipelineFakeProvider(
        entities=ExtractedEntities(
            tour="Ephesus",
            travel_date="2026-09-10",
            adults=2,
            children=1,
            cruise_ship="Equinox",
            hotel="Korumar",
            pickup_location="Port",
            preferred_language="English",
        )
    )
    result = run(
        ConversationPipelineService(provider), ConversationState(), "I want to book"
    )
    assert result.tour == "Ephesus"
    assert result.travel_date == date(2026, 9, 10)
    assert result.adults == 2
    assert result.children == 1
    assert result.cruise_ship == "Equinox"
    assert result.hotel == "Korumar"
    assert result.pickup_location == "Port"
    assert result.preferred_language == "English"


def test_repeated_calls_deterministic() -> None:
    provider1 = PipelineFakeProvider(entities=ExtractedEntities(adults=2))
    provider2 = PipelineFakeProvider(entities=ExtractedEntities(adults=2))
    state = ConversationState()
    message = "I want to book"
    first = run(ConversationPipelineService(provider1), state, message)
    second = run(ConversationPipelineService(provider2), state, message)
    assert first == second


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    run(
        ConversationPipelineService(PipelineFakeProvider()),
        ConversationState(),
        "hello",
    )
    assert dict(os.environ) == snapshot

