from datetime import datetime, timezone

from app.models.message import NormalizedMessage, TestMessageRequest


def normalize_test_message(payload: TestMessageRequest) -> NormalizedMessage:
    """Convert a test request into the internal inbound-message representation."""
    return NormalizedMessage(
        customer_phone=payload.sender,
        customer_name=payload.name,
        message=payload.message,
        source="test",
        received_at=datetime.now(timezone.utc),
    )
