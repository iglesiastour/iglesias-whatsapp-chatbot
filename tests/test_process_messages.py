from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.openrouter_client import OpenRouterError


client = TestClient(app)
URL = "/api/v1/messages/process"


def test_successful_ai_response() -> None:
    with patch(
        "app.routes.messages.generate_reply",
        new=AsyncMock(return_value="Hello Maria"),
    ) as generate:
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

    generate.assert_awaited_once_with("Hello")


def test_ai_service_failure() -> None:
    with patch(
        "app.routes.messages.generate_reply",
        new=AsyncMock(side_effect=OpenRouterError("provider failed")),
    ):
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
    with patch(
        "app.routes.messages.generate_reply",
        new=AsyncMock(return_value="Hello"),
    ) as generate:
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

    generate.assert_awaited_once_with("Hello")


def test_message_is_trimmed_before_ai_call() -> None:
    with patch(
        "app.routes.messages.generate_reply",
        new=AsyncMock(return_value="Hello"),
    ) as generate:
        response = client.post(
            URL,
            json={
                "from": "  +905551112233  ",
                "name": "  Maria  ",
                "message": "  Hello  ",
            },
        )

    assert response.status_code == 200
    generate.assert_awaited_once_with("Hello")