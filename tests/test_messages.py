from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "iglesias-whatsapp-chatbot",
    }


def test_valid_message_is_normalized_and_trimmed() -> None:
    response = client.post(
        "/api/v1/messages/test",
        json={
            "from": "  +905551112233  ",
            "name": "  Maria  ",
            "message": "  I want an Ephesus tour  ",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "data": {
            "customer_phone": "+905551112233",
            "customer_name": "Maria",
            "message": "I want an Ephesus tour",
            "source": "test",
        },
    }


def test_missing_message_is_rejected() -> None:
    response = client.post(
        "/api/v1/messages/test",
        json={"from": "+905551112233", "name": "Maria"},
    )

    assert response.status_code == 422


def test_empty_message_is_rejected() -> None:
    response = client.post(
        "/api/v1/messages/test",
        json={"from": "+905551112233", "message": "   "},
    )

    assert response.status_code == 422


def test_missing_sender_is_rejected() -> None:
    response = client.post(
        "/api/v1/messages/test",
        json={"name": "Maria", "message": "I want an Ephesus tour"},
    )

    assert response.status_code == 422
