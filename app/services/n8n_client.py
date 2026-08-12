import logging
from collections.abc import Mapping
from typing import Any

import httpx

from app.models.message import NormalizedMessage


logger = logging.getLogger(__name__)


class N8NError(Exception):
    """Base exception for controlled automation-service failures."""


class N8NNotConfiguredError(N8NError):
    """Raised when no automation webhook is configured."""


class N8NTimeoutError(N8NError):
    """Raised when the automation service does not respond in time."""


class N8NConnectionError(N8NError):
    """Raised when the automation service cannot be reached."""


class N8NResponseError(N8NError):
    """Raised when the automation service returns an unusable response."""


def mask_phone(phone: str) -> str:
    """Mask the middle of a phone number before writing it to logs."""
    if len(phone) <= 4:
        return "*" * len(phone)
    visible_prefix = min(5, len(phone) - 2)
    return f"{phone[:visible_prefix]}{'*' * (len(phone) - visible_prefix - 2)}{phone[-2:]}"


def parse_n8n_reply(response_data: Any) -> str:
    """Extract a non-empty assistant reply from a supported n8n response."""
    if not isinstance(response_data, Mapping):
        raise N8NResponseError("Automation service returned an invalid response.")

    for field in ("reply", "output", "message"):
        value = response_data.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    raise N8NResponseError("Automation service response did not contain a reply.")


async def forward_to_n8n(
    message: NormalizedMessage,
    webhook_url: str,
    timeout_seconds: float,
) -> str:
    """Forward one normalized message to n8n and return its parsed reply."""
    if not webhook_url.strip():
        raise N8NNotConfiguredError("Automation service is not configured.")

    payload = {
        "from": message.customer_phone,
        "name": message.customer_name,
        "message": message.message,
    }
    masked_phone = mask_phone(message.customer_phone)
    logger.info("Forwarding customer message to automation service phone=%s", masked_phone)

    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(webhook_url, json=payload)
    except httpx.TimeoutException as exc:
        logger.warning("Automation service timed out phone=%s", masked_phone)
        raise N8NTimeoutError("Automation service timed out.") from exc
    except httpx.RequestError as exc:
        logger.error("Automation service connection failed phone=%s error_type=%s", masked_phone, type(exc).__name__)
        raise N8NConnectionError("Automation service is unavailable.") from exc

    logger.info("Automation service responded status=%s phone=%s", response.status_code, masked_phone)
    if not response.is_success:
        raise N8NResponseError("Automation service returned an error response.")

    try:
        response_data = response.json()
    except ValueError as exc:
        raise N8NResponseError("Automation service returned an invalid response.") from exc

    return parse_n8n_reply(response_data)
