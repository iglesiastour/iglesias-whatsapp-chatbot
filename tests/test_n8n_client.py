import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.models.message import NormalizedMessage
from app.services.n8n_client import (
    N8NConnectionError,
    N8NNotConfiguredError,
    N8NResponseError,
    N8NTimeoutError,
    forward_to_n8n,
    parse_n8n_reply,
)


@pytest.fixture
def normalized_message() -> NormalizedMessage:
    return NormalizedMessage(
        customer_phone="+905551112233",
        customer_name="Maria",
        message="Hello",
        source="test",
        received_at=datetime.now(timezone.utc),
    )


@pytest.mark.parametrize(
    ("body", "expected"),
    [({"reply": "Hello"}, "Hello"), ({"output": "Hi"}, "Hi"), ({"message": "Merhaba"}, "Merhaba")],
)
def test_parse_n8n_reply_supports_known_formats(body: object, expected: str) -> None:
    assert parse_n8n_reply(body) == expected


def test_parse_n8n_reply_rejects_unknown_format() -> None:
    with pytest.raises(N8NResponseError):
        parse_n8n_reply({"data": {"reply": "not a supported shape"}})


def test_missing_webhook_url_is_rejected(normalized_message: NormalizedMessage) -> None:
    with pytest.raises(N8NNotConfiguredError):
        asyncio.run(forward_to_n8n(normalized_message, "", 20))


def test_timeout_is_converted(normalized_message: NormalizedMessage) -> None:
    request = httpx.Request("POST", "https://example.test/webhook")
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.ReadTimeout("timeout", request=request))):
        with pytest.raises(N8NTimeoutError):
            asyncio.run(forward_to_n8n(normalized_message, "https://example.test/webhook", 20))


def test_connection_failure_is_converted(normalized_message: NormalizedMessage) -> None:
    request = httpx.Request("POST", "https://example.test/webhook")
    with patch("httpx.AsyncClient.post", new=AsyncMock(side_effect=httpx.ConnectError("failed", request=request))):
        with pytest.raises(N8NConnectionError):
            asyncio.run(forward_to_n8n(normalized_message, "https://example.test/webhook", 20))


def test_non_2xx_response_is_rejected(normalized_message: NormalizedMessage) -> None:
    response = httpx.Response(500, request=httpx.Request("POST", "https://example.test/webhook"))
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=response)):
        with pytest.raises(N8NResponseError):
            asyncio.run(forward_to_n8n(normalized_message, "https://example.test/webhook", 20))
