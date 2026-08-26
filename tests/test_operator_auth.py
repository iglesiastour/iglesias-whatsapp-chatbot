"""Tests for the operator authentication boundary (no DB/network)."""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security.operator_auth import require_operator_credentials

client = TestClient(app)
TOKEN = "unit-test-operator-token"


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", TOKEN)


def _auth(value: str | None = None, header: bool = True):
    headers = {"Authorization": value} if value is not None else {}
    return client.get(
        "/api/v1/operator/handoffs/12345678-1234-5678-1234-567812345678",
        headers=headers if header else None,
    )


def test_correct_bearer_token_passes_dependency():
    require_operator_credentials(f"Bearer {TOKEN}")


# --- HTTP-level behavior (404 expected after auth passes; auth failure codes asserted) --


def test_missing_authorization_header_rejected_401():
    with patch(
        "app.routes.operator_handoffs.get_handoff_repository", MagicMock()
    ):
        response = _auth(header=False)
    assert response.status_code == 401
    assert response.json() == {"detail": "Operator authentication required."}


def test_wrong_token_rejected_401():
    response = _auth("Bearer wrong-token")
    assert response.status_code == 401


def test_empty_bearer_token_rejected_401():
    assert _auth("Bearer ").status_code == 401
    assert _auth("Bearer    ").status_code == 401


def test_basic_scheme_rejected_401():
    assert _auth(f"Basic {TOKEN}").status_code == 401


def test_malformed_bearer_header_rejected_401():
    assert _auth("Bearer").status_code == 401
    assert _auth("bearer-no-space").status_code == 401


def test_extra_token_malformed_header_rejected_401():
    assert _auth(f"Bearer {TOKEN} extra").status_code == 401


def test_unconfigured_server_token_fails_closed_503(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", "")
    response = _auth(f"Bearer {TOKEN}")
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Operator authentication is unavailable."
    }


def test_whitespace_only_configured_token_fails_closed_503(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", "   ")
    response = _auth(f"Bearer {TOKEN}")
    assert response.status_code == 503


def test_configured_token_never_in_response_body():
    for value in ("wrong-token", f"Bearer {TOKEN}"):
        response = _auth(value)
        assert TOKEN not in response.text


def test_incorrect_supplied_token_not_echoed():
    supplied = "supplied-secret-value"
    response = _auth(f"Bearer {supplied}")
    assert supplied not in response.text


def test_timing_safe_comparison_used():
    source = inspect.getsource(require_operator_credentials)
    assert "secrets.compare_digest" in source


def test_unauthorized_detail_is_static_and_does_not_vary_by_failure_mode():
    responses = [
        _auth(header=False).json(),
        _auth("Basic x").json(),
        _auth("Bearer ").json(),
        _auth("Bearer nope").json(),
    ]
    assert all(r == {"detail": "Operator authentication required."} for r in responses)


def test_auth_module_has_no_db_network_ai_imports():
    import app.security.operator_auth as module

    source = inspect.getsource(module)
    for forbidden in (
        "psycopg",
        "database_connection",
        "get_handoff_repository",
        "OpenRouterProvider",
        "SafeAIService",
        "httpx",
        "requests",
    ):
        assert forbidden not in source, forbidden


def test_case_insensitive_bearer_scheme_accepted():
    # Scheme comparison is case-insensitive; token itself is exact.
    with patch(
        "app.routes.operator_handoffs.get_handoff_repository"
    ) as factory:
        factory.return_value.get.return_value = SimpleNamespace(id="x")
        response = _auth(f"bearer {TOKEN}")
    assert response.status_code != 401
