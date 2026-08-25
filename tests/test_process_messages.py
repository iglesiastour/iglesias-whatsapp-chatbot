from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.ai.base import AIProvider, AIProviderError


client = TestClient(app)
URL = "/api/v1/messages/process"


def _mock_provider(return_value: str | None = None, side_effect: Exception | None = None):
    provider = AsyncMock(spec=AIProvider)
    if side_effect is not None:
        provider.generate_reply.side_effect = side_effect
    else:
        provider.generate_reply.return_value = return_value
    return patch("app.routes.messages.get_ai_provider", return_value=provider)


def test_successful_ai_response() -> None:
    with _mock_provider(return_value="Hello Maria") as factory:
        provider = factory.return_value
        response = client.post(
            URL,
            json={
                "from": "+905551112233",
                "name": "Maria",
                "message": "Hello",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {
            "customer_phone": "+905551112233",
            "reply": "Hello Maria",
        },
    }

    provider.generate_reply.assert_awaited_once_with("Hello")


def test_ai_service_failure() -> None:
    with _mock_provider(side_effect=AIProviderError("provider failed")):
        response = client.post(
            URL,
            json={
                "from": "+905551112233",
                "message": "Hello",
            },
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "AI service is unavailable.",
    }


def test_invalid_incoming_payload() -> None:
    response = client.post(
        URL,
        json={
            "from": "+905551112233",
            "message": "   ",
        },
    )

    assert response.status_code == 422


def test_name_missing_but_message_valid() -> None:
    with _mock_provider(return_value="Hello") as factory:
        provider = factory.return_value
        response = client.post(
            URL,
            json={
                "from": "+905551112233",
                "message": "Hello",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {
            "customer_phone": "+905551112233",
            "reply": "Hello",
        },
    }

    provider.generate_reply.assert_awaited_once_with("Hello")


def test_message_is_trimmed_before_ai_call() -> None:
    with _mock_provider(return_value="Hello") as factory:
        provider = factory.return_value
        response = client.post(
            URL,
            json={
                "from": "  +905551112233  ",
                "name": "  Maria  ",
                "message": "  Hello  ",
            },
        )

    assert response.status_code == 200
    provider.generate_reply.assert_awaited_once_with("Hello")
