import logging

from fastapi import APIRouter, HTTPException, status


from app.models.message import (
    MessageData,
    ProcessMessageData,
    ProcessMessageResponse,
    TestMessageRequest,
    TestMessageResponse,
)
from app.services.message_normalizer import normalize_test_message
from app.services.ai.base import AIProviderError
from app.services.ai.provider import get_ai_provider
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
    """Normalize a customer message and generate a safe AI reply."""
    normalized = normalize_test_message(payload)

    provider = get_ai_provider()
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