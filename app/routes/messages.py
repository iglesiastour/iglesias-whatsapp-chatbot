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
)
from app.services.conversation_pipeline_service import ConversationPipelineService
from app.services.safe_ai_service import SafeAIService


router = APIRouter(prefix="/messages", tags=["messages"])
logger = logging.getLogger(__name__)


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

    service = SafeAIService(provider)

    try:
        result = await service.generate_reply(normalized.message)
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