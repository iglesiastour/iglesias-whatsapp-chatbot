import logging

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.models.message import (
    MessageData,
    ProcessMessageData,
    ProcessMessageResponse,
    TestMessageRequest,
    TestMessageResponse,
)
from app.services.message_normalizer import normalize_test_message
from app.services.n8n_client import (
    N8NConnectionError,
    N8NNotConfiguredError,
    N8NResponseError,
    N8NTimeoutError,
    forward_to_n8n,
)


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
    """Normalize a customer message and forward it to the existing n8n workflow."""
    normalized = normalize_test_message(payload)
    try:
        reply = await forward_to_n8n(
            normalized,
            webhook_url=settings.n8n_webhook_url,
            timeout_seconds=settings.n8n_timeout_seconds,
        )
    except N8NNotConfiguredError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Automation service is not configured.",
        ) from None
    except N8NTimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Automation service timed out.",
        ) from None
    except (N8NConnectionError, N8NResponseError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Automation service is unavailable.",
        ) from None

    return ProcessMessageResponse(
        success=True,
        data=ProcessMessageData(
            customer_phone=normalized.customer_phone,
            reply=reply,
        ),
    )
