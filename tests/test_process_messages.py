from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.services.n8n_client import (
    N8NConnectionError,
    N8NNotConfiguredError,
    N8NResponseError,
    N8NTimeoutError,
)


client = TestClient(app)
URL = "/api/v1/messages/process"


def test_successful_n8n_response() -> None:
    with patch("app.routes.messages.forward_to_n8n", new=AsyncMock(return_value="Hello Maria")) as forward:
        response = client.post(
            URL,
            json={"from": "+905551112233", "name": "Maria", "message": "Hello"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {"customer_phone": "+905551112233", "reply": "Hello Maria"},
    }
    normalized = forward.await_args.args[0]
    assert normalized.customer_phone == "+905551112233"
    assert normalized.message == "Hello"


def test_n8n_url_missing() -> None:
    with patch(
        "app.routes.messages.forward_to_n8n",
        new=AsyncMock(side_effect=N8NNotConfiguredError),
    ):
        response = client.post(URL, json={"from": "+905551112233", "message": "Hello"})

    assert response.status_code == 503
    assert response.json() == {"detail": "Automation service is not configured."}


def test_n8n_timeout() -> None:
    with patch("app.routes.messages.forward_to_n8n", new=AsyncMock(side_effect=N8NTimeoutError)):
        response = client.post(URL, json={"from": "+905551112233", "message": "Hello"})

    assert response.status_code == 504
    assert response.json() == {"detail": "Automation service timed out."}


def test_n8n_connection_failure() -> None:
    with patch("app.routes.messages.forward_to_n8n", new=AsyncMock(side_effect=N8NConnectionError)):
        response = client.post(URL, json={"from": "+905551112233", "message": "Hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Automation service is unavailable."}


def test_n8n_non_2xx_response() -> None:
    with patch("app.routes.messages.forward_to_n8n", new=AsyncMock(side_effect=N8NResponseError)):
        response = client.post(URL, json={"from": "+905551112233", "message": "Hello"})

    assert response.status_code == 502


def test_invalid_incoming_payload() -> None:
    response = client.post(URL, json={"from": "+905551112233", "message": "   "})

    assert response.status_code == 422


def test_name_missing_but_message_valid() -> None:
    with patch("app.routes.messages.forward_to_n8n", new=AsyncMock(return_value="Hello")) as forward:
        response = client.post(URL, json={"from": "+905551112233", "message": "Hello"})

    assert response.status_code == 200
    assert forward.await_args.args[0].customer_name is None
