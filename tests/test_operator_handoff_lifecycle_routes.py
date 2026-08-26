"""Phase 7 Step 2: protected handoff lifecycle transition route tests."""

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.db.connection import DatabaseNotConfiguredError
from app.main import app
from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.repositories.provider import RepositoryConfigurationError

client = TestClient(app)
URL = "/api/v1/operator/handoffs"
HANDOFF_ID = "12345678-1234-5678-1234-567812345678"
TOKEN = "lifecycle-test-operator-token"


def _state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
        booking_stage=BookingStage.READY_FOR_REVIEW,
        needs_human=True,
    )


def _persisted(status: HandoffStatus) -> PersistedHandoff:
    return PersistedHandoff(
        id=UUID(HANDOFF_ID),
        idempotency_key="k" * 64,
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        status=status,
        conversation_state=_state(),
    )


class SpyRepo(InMemoryHandoffRepository):
    def __init__(self):
        super().__init__()
        self.update_calls = 0

    def update_status(self, handoff_id, status):  # type: ignore[override]
        self.update_calls += 1
        return super().update_status(handoff_id, status)


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", TOKEN)


def _seeded(status: HandoffStatus):
    repo = SpyRepo()
    handoff = _persisted(status=status)
    repo._store[handoff.id] = handoff.model_copy(deep=True)
    repo._by_key[handoff.idempotency_key] = handoff.id
    factory_calls = {"count": 0}

    def factory():
        factory_calls["count"] += 1
        return repo

    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", side_effect=factory
    )
    return {"repo": repo, "patcher": patcher, "factory_calls": factory_calls}


def _patch_status(
    target: HandoffStatus,
    handoff_id: str = HANDOFF_ID,
    headers=None,
    payload=None,
):
    body = payload if payload is not None else {"status": target.value}
    return client.patch(
        f"{URL}/{handoff_id}/status",
        json=body,
        headers=headers or {"Authorization": f"Bearer {TOKEN}"},
    )


# --- AUTH ----------------------------------------------------------------------


def test_missing_auth_rejected_401():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = client.patch(
            f"{URL}/{HANDOFF_ID}/status", json={"status": "in_review"}
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Operator authentication required."}


def test_wrong_auth_rejected_401():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = _patch_status(
            HandoffStatus.IN_REVIEW, headers={"Authorization": "Bearer nope"}
        )

    assert response.status_code == 401


def test_auth_unconfigured_fails_closed_503(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", "")
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Operator authentication is unavailable."
    }


# --- SUCCESS TRANSITIONS ---------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (HandoffStatus.PENDING, HandoffStatus.IN_REVIEW),
        (HandoffStatus.PENDING, HandoffStatus.RESOLVED),
        (HandoffStatus.PENDING, HandoffStatus.CANCELLED),
        (HandoffStatus.IN_REVIEW, HandoffStatus.RESOLVED),
        (HandoffStatus.IN_REVIEW, HandoffStatus.CANCELLED),
    ],
)
def test_allowed_transitions_return_200(current, target):
    ctx = _seeded(current)
    with ctx["patcher"]:
        response = _patch_status(target)

    assert response.status_code == 200
    assert response.json()["status"] == target.value


# --- SAME STATUS ------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        HandoffStatus.PENDING,
        HandoffStatus.IN_REVIEW,
        HandoffStatus.RESOLVED,
        HandoffStatus.CANCELLED,
    ],
)
def test_same_status_is_idempotent_200(status: HandoffStatus):
    ctx = _seeded(status)
    with ctx["patcher"]:
        response = _patch_status(status)

    assert response.status_code == 200
    assert response.json()["status"] == status.value


@pytest.mark.parametrize(
    "status",
    [
        HandoffStatus.PENDING,
        HandoffStatus.IN_REVIEW,
        HandoffStatus.RESOLVED,
        HandoffStatus.CANCELLED,
    ],
)
def test_same_status_skips_update_status_call(status: HandoffStatus):
    ctx = _seeded(status)
    with ctx["patcher"]:
        _patch_status(status)

    assert ctx["repo"].update_calls == 0


# --- INVALID TRANSITIONS ------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (HandoffStatus.IN_REVIEW, HandoffStatus.PENDING),
        (HandoffStatus.RESOLVED, HandoffStatus.PENDING),
        (HandoffStatus.RESOLVED, HandoffStatus.IN_REVIEW),
        (HandoffStatus.RESOLVED, HandoffStatus.CANCELLED),
        (HandoffStatus.CANCELLED, HandoffStatus.PENDING),
        (HandoffStatus.CANCELLED, HandoffStatus.IN_REVIEW),
        (HandoffStatus.CANCELLED, HandoffStatus.RESOLVED),
    ],
)
def test_invalid_transitions_return_409(current, target):
    ctx = _seeded(current)
    with ctx["patcher"]:
        response = _patch_status(target)

    assert response.status_code == 409
    assert response.json() == {"detail": "Invalid handoff status transition."}


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (HandoffStatus.IN_REVIEW, HandoffStatus.PENDING),
        (HandoffStatus.RESOLVED, HandoffStatus.PENDING),
        (HandoffStatus.CANCELLED, HandoffStatus.RESOLVED),
    ],
)
def test_invalid_transition_does_not_mutate_stored_object(current, target):
    ctx = _seeded(current)
    with ctx["patcher"]:
        _patch_status(target)

    stored = ctx["repo"].get(UUID(HANDOFF_ID))
    assert stored.status is current


# --- VALIDATION -----------------------------------------------------------------


def test_malformed_uuid_rejected_422_without_repository_call():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = client.patch(
            f"{URL}/not-a-uuid/status",
            json={"status": "in_review"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 422
    assert ctx["factory_calls"]["count"] == 0


def test_invalid_status_enum_rejected_422():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = _patch_status(
            HandoffStatus.PENDING, payload={"status": "super_urgent"}
        )

    assert response.status_code == 422


def test_missing_status_field_rejected_422():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = client.patch(
            f"{URL}/{HANDOFF_ID}/status",
            json={},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "extra_payload",
    [
        {"status": "resolved", "reason": "booking_review"},
        {"status": "resolved", "booking_stage": "confirmed"},
        {"status": "resolved", "conversation_state": {}},
        {"status": "resolved", "idempotency_key": "k" * 64},
        {"status": "resolved", "customer_phone": "+900000"},
        {"status": "resolved", "notes": "hello"},
        {"status": "resolved", "tour": "Ephesus"},
    ],
)
def test_extra_fields_forbidden_422(extra_payload):
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = client.patch(
            f"{URL}/{HANDOFF_ID}/status",
            json=extra_payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 422


# --- NOT FOUND / STORAGE -----------------------------------------------------------


def test_missing_handoff_returns_safe_404():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = _patch_status(
            HandoffStatus.IN_REVIEW,
            handoff_id="00000000-0000-0000-0000-000000000000",
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Handoff not found."}
    assert HANDOFF_ID not in response.text


def test_repository_configuration_error_safe_500():
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository",
        side_effect=RepositoryConfigurationError("postgres"),
    )
    with patcher:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Human review service is not configured correctly."
    }
    assert "postgres" not in response.text.lower()


def test_database_not_configured_error_safe_503():
    repo = MagicMock()
    repo.get.side_effect = DatabaseNotConfiguredError("postgresql://secret")
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", return_value=repo
    )
    with patcher:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 503
    assert response.json() == {"detail": "Human review service is unavailable."}


def test_unexpected_persistence_error_safe_503():
    repo = MagicMock()
    repo.get.return_value = _persisted(HandoffStatus.PENDING)
    repo.update_status.side_effect = RuntimeError("boom")
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", return_value=repo
    )
    with patcher:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 503


def test_raw_sql_error_and_db_url_not_leaked():
    leaked_url = "postgresql://user:hunter2@neon.example/db"
    repo = MagicMock()
    repo.get.return_value = _persisted(HandoffStatus.PENDING)
    repo.update_status.side_effect = RuntimeError(
        f'psycopg: duplicate key ... {leaked_url} DETAIL idempotency_key=("k" * 64)'
    )
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", return_value=repo
    )
    with patcher:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 503
    assert leaked_url not in response.text
    assert "hunter2" not in response.text
    assert "k" * 64 not in response.text


# --- RESPONSE / DATA INTEGRITY ------------------------------------------------------


def test_response_matches_safe_review_contract_after_transition():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        body = _patch_status(HandoffStatus.IN_REVIEW).json()

    assert set(body) == {
        "handoff_id",
        "customer_phone",
        "customer_name",
        "reason",
        "status",
        "intent",
        "booking_stage",
        "needs_human",
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    }
    assert body["status"] == "in_review"
    assert body["booking_stage"] == "ready_for_review"
    assert body["reason"] == "booking_review"
    assert body["customer_phone"] == "+90555 111 2233"
    assert body["customer_name"] == "Mehmet Cam"
    assert body["tour"] == "Ephesus tour"
    assert body["travel_date"] == "2026-09-10"
    assert body["adults"] == 2
    assert body["children"] == 1
    assert body["cruise_ship"] == "Equinox"
    assert body["hotel"] == "Korumar"
    assert body["pickup_location"] == "Port"
    assert body["preferred_language"] == "English"


def test_sensitive_fields_absent_from_patch_response():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    text = response.text.lower()
    for forbidden in (
        "idempotency_key",
        "created_at",
        "updated_at",
        "provider",
        "model",
        "api_key",
        "database_url",
        "prompt",
        "reasoning",
        "transcript",
    ):
        assert forbidden not in text


# --- READ / LIFECYCLE ISOLATION ------------------------------------------------------


def test_get_remains_read_only_after_patch_addition():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        assert _patch_status(HandoffStatus.IN_REVIEW).status_code == 200

        get_spy = MagicMock(wraps=ctx["repo"].get)
        update_spy = MagicMock(wraps=ctx["repo"].update_status)
        ctx["repo"].get = get_spy  # type: ignore[method-assign]
        ctx["repo"].update_status = update_spy  # type: ignore[method-assign]

        response = client.get(
            f"{URL}/{HANDOFF_ID}", headers={"Authorization": f"Bearer {TOKEN}"}
        )

    assert response.status_code == 200
    assert response.json()["status"] == "in_review"
    assert get_spy.call_count == 1
    update_spy.assert_not_called()


def test_repeated_get_does_not_change_status():
    ctx = _seeded(HandoffStatus.IN_REVIEW)
    with ctx["patcher"]:
        for _ in range(3):
            response = client.get(
                f"{URL}/{HANDOFF_ID}",
                headers={"Authorization": f"Bearer {TOKEN}"},
            )
            assert response.json()["status"] == "in_review"

    assert ctx["repo"].update_calls == 0


def test_customer_message_route_contains_no_lifecycle_transition():
    import inspect

    import app.routes.messages as route_module

    source = inspect.getsource(route_module)
    for forbidden in (
        "HandoffLifecycleService",
        ".transition(",
        ".update_status(",
    ):
        assert forbidden not in source, forbidden


def test_safeai_and_openrouter_do_not_import_lifecycle():
    import inspect

    import app.services.ai.openrouter as openrouter_module
    import app.services.safe_ai_service as safe_ai_module

    for module_src in (
        inspect.getsource(safe_ai_module),
        inspect.getsource(openrouter_module),
    ):
        assert "handoff_lifecycle_service" not in module_src
        assert "HandoffLifecycleService" not in module_src


# --- ARCHITECTURE --------------------------------------------------------------------


def test_route_uses_factory_and_lifecycle_service():
    import inspect

    import app.routes.operator_handoffs as module

    source = inspect.getsource(module)
    assert "get_handoff_repository()" in source
    assert "HandoffLifecycleService(" in source
    assert "lifecycle.transition(" in source


def test_one_repository_object_per_patch_request():
    ctx = _seeded(HandoffStatus.PENDING)
    with ctx["patcher"]:
        _patch_status(HandoffStatus.IN_REVIEW)

    assert ctx["factory_calls"]["count"] == 1
