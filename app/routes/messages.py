import logging

from fastapi import APIRouter, HTTPException, status


from app.models.message import (
    MessageData,
    ProcessMessageData,
    ProcessMessageResponse,
    TestMessageRequest,
    TestMessageResponse,
)
from app.db.connection import DatabaseNotConfiguredError
from app.services.message_normalizer import normalize_test_message
from app.security.prompt_guard import inspect_prompt
from app.services.ai.base import AIProviderError
from app.services.ai.provider import get_ai_provider
from app.repositories.provider import (
    RepositoryConfigurationError,
    get_conversation_repository,
    get_handoff_repository,
)
from app.services.conversation_pipeline_service import ConversationPipelineService
from app.prompts.conversation_context import build_conversation_context
from app.services.handoff_service import HandoffService
from app.services.safe_ai_service import SafeAIService
from app.services.safety_fallback_service import SafetyFallbackContext


router = APIRouter(prefix="/messages", tags=["messages"])
logger = logging.getLogger(__name__)


def _ensure_handoff(
    customer_phone: str,
    state,
    customer_name: str | None,
) -> None:
    """Ensure a human-review handoff for persisted state (lazy factory use).

    Narrow error boundary: only handoff construction/persistence failures are
    mapped here; AI/validation errors are never caught.
    """
    try:
        handoff_repository = get_handoff_repository()
    except RepositoryConfigurationError:
        # Do not echo invalid backend/config values.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review service is not configured correctly.",
        ) from None

    try:
        handoff_service = HandoffService(handoff_repository)
        handoff_service.ensure_handoff(
            customer_phone=customer_phone,
            state=state,
            customer_name=customer_name,
        )
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Handoff persistence failed.")
        # Safe detail only: no SQL/UUID/key/backend leakage.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None


@router.post("/test", response_model=TestMessageResponse)
async def test_message(payload: TestMessageRequest) -> TestMessageResponse:
    """Validate and normalize a simulated incoming WhatsApp message."""
    normalized = normalize_test_message(payload)
    return TestMessageResponse(
        success=True,
        data=MessageData(
            customer_phone=normalized.customer_phone,
            customer_name=normalized.customer_name,
            message=normalized.message,
            source=normalized.source,
        ),
    )


@router.post("/process", response_model=ProcessMessageResponse)
async def process_message(payload: TestMessageRequest) -> ProcessMessageResponse:
    """Normalize a customer message, update conversation state, and reply."""
    normalized = normalize_test_message(payload)

    try:
        repository = get_conversation_repository()
    except RepositoryConfigurationError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation storage is not configured correctly.",
        ) from None

    provider = get_ai_provider()

    # Prompt-injection messages must never reach entity extraction or alter
    # stored state; SafeAIService handles the safe conversational redirect.
    prompt_result = inspect_prompt(normalized.message)

    conversation_context: str | None = None

    if prompt_result.is_safe:
        try:
            current_state = repository.get(normalized.customer_phone)
        except DatabaseNotConfiguredError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation storage is unavailable.",
            ) from None

        pipeline = ConversationPipelineService(provider)

        try:
            updated_state = await pipeline.process_message(
                current_state,
                normalized.message,
            )
        except AIProviderError:
            # Do not save a partially updated state.
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="AI service is unavailable.",
            ) from None

        try:
            # State was successfully interpreted; keep it even if reply
            # generation fails below (documented phase behavior).
            repository.save(normalized.customer_phone, updated_state)
        except DatabaseNotConfiguredError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation storage is unavailable.",
            ) from None

        # State is persisted; now ensure the human-review handoff from the
        # UPDATED state. Idempotency and reason rules live in HandoffService.
        _ensure_handoff(
            customer_phone=normalized.customer_phone,
            state=updated_state,
            customer_name=normalized.customer_name,
        )

        conversation_context = build_conversation_context(
            updated_state,
            customer_name=normalized.customer_name,
        )

    service = SafeAIService(provider)

    known_tour = updated_state.tour if prompt_result.is_safe else None
    booking_stage = updated_state.booking_stage if prompt_result.is_safe else None

    fallback_context: SafetyFallbackContext | None = None
    if prompt_result.is_safe:
        fallback_context = SafetyFallbackContext(
            tour=updated_state.tour,
            travel_date=updated_state.travel_date,
            adults=updated_state.adults,
            children=updated_state.children,
            cruise_ship=updated_state.cruise_ship,
            hotel=updated_state.hotel,
            pickup_location=updated_state.pickup_location,
            preferred_language=updated_state.preferred_language,
            booking_stage=updated_state.booking_stage,
            requires_human=updated_state.requires_human,
            missing_booking_fields=updated_state.missing_booking_fields(),
        )

    try:
        result = await service.generate_reply(
            normalized.message,
            conversation_context=conversation_context,
            known_tour=known_tour,
            booking_stage=booking_stage,
            fallback_context=fallback_context,
        )
    except AIProviderError:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service is unavailable.",
        ) from None

    return ProcessMessageResponse(
        success=True,
        data=ProcessMessageData(
            customer_phone=normalized.customer_phone,
            reply=result.reply,
        ),
)