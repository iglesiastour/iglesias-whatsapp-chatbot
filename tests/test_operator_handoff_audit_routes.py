"""Phase 7 Step 4: handoff status audit-trail route tests.

Covers the append-only audit lifecycle orchestrated through the operator PATCH
status endpoint and the read-only GET /audit endpoint. No PII, auth tokens,
booking details, or AI internals are exposed.
"""

from datetime import date
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff
from app.models.handoff_audit import HandoffAuditEvent  # noqa: F401
from app.repositories.handoff_audit_repository import HandoffAuditError
from app.repositories.in_memory_handoff_audit_repository import (
    InMemoryHandoffAuditRepository,
)
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository

client = TestClient(app)
URL = "/api/v1/operator/handoffs"
HANDOFF_ID = "12345678-1234-5678-1234-567812345678"
TOKEN = "audit-test-operator-token"


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


class SpyHandoffRepo(InMemoryHandoffRepository):
    def __init__(self):
        super().__init__()
        self.update_calls = 0

    def update_status(self, handoff_id, status):  # type: ignore[override]
        self.update_calls += 1
        return super().update_status(handoff_id, status)


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", TOKEN)


class _PatchersCtx:
    def __init__(self, ctx):
        self._ctx = ctx

    def __enter__(self):
        for p in self._ctx["patchers"]:
            p.start()
        return self._ctx

    def __exit__(self, *exc):
        for p in self._ctx["patchers"]:
            p.stop()
        return False


def _seed(handoff: PersistedHandoff) -> _PatchersCtx:
    handoff_repo = SpyHandoffRepo()
    handoff_repo._store[handoff.id] = handoff.model_copy(deep=True)
    handoff_repo._by_key[handoff.idempotency_key] = handoff.id
    audit_repo = InMemoryHandoffAuditRepository()

    calls = {"handoff_factory": 0, "audit_factory": 0}

    def handoff_factory():
        calls["handoff_factory"] += 1
        return handoff_repo

    def audit_factory():
        calls["audit_factory"] += 1
        return audit_repo

    patcher_audit = patch(
        "app.routes.operator_handoffs.get_handoff_audit_repository",
        side_effect=audit_factory,
    )
    patcher_handoff = patch(
        "app.routes.operator_handoffs.get_handoff_repository",
        side_effect=handoff_factory,
    )
    ctx = {
        "handoff": handoff_repo,
        "audit": audit_repo,
        "calls": calls,
        "patchers": (patcher_audit, patcher_handoff),
    }
    return _PatchersCtx(ctx)


def _auth(headers=None) -> dict:
    return headers or {"Authorization": f"Bearer {TOKEN}"}


def _patch_status(status, headers=None):
    return client.patch(
        f"{URL}/{HANDOFF_ID}/status",
        json={"status": status.value},
        headers=_auth(headers),
    )


def _get_audit(headers=None, handoff_id=None):
    hid = handoff_id or HANDOFF_ID
    return client.get(f"{URL}/{hid}/audit", headers=_auth(headers))


# --- PATCH creates audit on real status change ----------------------------------


def test_patch_real_change_creates_one_audit():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 200
    events = ctx["audit"].list_for_handoff(UUID(HANDOFF_ID))
    assert len(events) == 1
    assert events[0].previous_status is HandoffStatus.PENDING
    assert events[0].new_status is HandoffStatus.IN_REVIEW


def test_patch_same_status_creates_no_audit():
    with _seed(_persisted(HandoffStatus.IN_REVIEW)) as ctx:
        response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 200
    assert ctx["audit"].list_for_handoff(UUID(HANDOFF_ID)) == []
    assert ctx["handoff"].update_calls == 0


def test_invalid_patch_creates_no_audit():
    with _seed(_persisted(HandoffStatus.IN_REVIEW)) as ctx:
        response = _patch_status(HandoffStatus.PENDING)

    assert response.status_code == 409
    assert ctx["audit"].list_for_handoff(UUID(HANDOFF_ID)) == []


# --- GET /audit endpoint ---------------------------------------------------------


def test_get_audit_authenticated():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        response = _get_audit()

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_get_audit_missing_auth_401():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        response = client.get(f"{URL}/{HANDOFF_ID}/audit")

    assert response.status_code == 401


def test_get_audit_auth_unconfigured_503(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", "")
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        response = _get_audit()

    assert response.status_code == 503
    assert response.json() == {"detail": "Operator authentication is unavailable."}


def test_get_audit_unknown_handoff_404():
    other = "87654321-4321-8765-4321-876543210000"
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        response = _get_audit(handoff_id=other)

    assert response.status_code == 404
    assert response.json() == {"detail": "Handoff not found."}


def test_get_audit_exact_contract():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        ctx["audit"].create_status_change(
            handoff_id=UUID(HANDOFF_ID),
            previous_status=HandoffStatus.PENDING,
            new_status=HandoffStatus.IN_REVIEW,
        )
        response = _get_audit()

    assert response.status_code == 200
    item = response.json()["items"][0]
    assert set(item) == {
        "id",
        "handoff_id",
        "action",
        "previous_status",
        "new_status",
        "created_at",
    }


def test_get_audit_no_pii_or_internals():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        ctx["audit"].create_status_change(
            handoff_id=UUID(HANDOFF_ID),
            previous_status=HandoffStatus.PENDING,
            new_status=HandoffStatus.IN_REVIEW,
        )
        body = _get_audit().text

    for forbidden in (
        "customer_phone",
        "customer_name",
        "tour",
        "booking_stage",
        "idempotency_key",
        "prompt",
        "provider",
        "model_name",
        "sql",
        "database_url",
        "Maria",
        "Ephesus",
    ):
        assert forbidden.lower() not in body.lower()


def test_get_audit_repeated_side_effect_free():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        first = _get_audit()
        second = _get_audit()

    assert first.status_code == 200
    assert first.json() == second.json()
    assert ctx["audit"].list_for_handoff(UUID(HANDOFF_ID)) == []


def test_get_audit_does_not_update_handoff():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        _get_audit()

    assert ctx["handoff"].update_calls == 0
    stored = ctx["handoff"].get(UUID(HANDOFF_ID))
    assert stored.status is HandoffStatus.PENDING


# --- Audit storage errors --------------------------------------------------------


def test_audit_storage_error_safe_503():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        erroring = MagicMock()
        erroring.list_for_handoff.side_effect = RuntimeError("corrupt psycopg state")
        with patch(
            "app.routes.operator_handoffs.get_handoff_audit_repository",
            return_value=erroring,
        ):
            response = _get_audit()

    assert response.status_code == 503
    assert response.json() == {"detail": "Human review service is unavailable."}
    assert "corrupt" not in response.text
    assert "psycopg" not in response.text


# --- Audit-failure-after-status-update semantics --------------------------------


def test_audit_failure_after_update_503_but_status_updated():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        failing_audit = MagicMock()
        failing_audit.create_status_change.side_effect = HandoffAuditError(
            "audit write failed"
        )
        with patch(
            "app.routes.operator_handoffs.get_handoff_audit_repository",
            return_value=failing_audit,
        ):
            response = _patch_status(HandoffStatus.IN_REVIEW)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Human review audit service is unavailable."
    }
    stored = ctx["handoff"].get(UUID(HANDOFF_ID))
    assert stored.status is HandoffStatus.IN_REVIEW
    assert ctx["handoff"].update_calls == 1


# --- Regression: GET / PATCH still work ------------------------------------------


def test_existing_get_and_patch_still_work():
    with _seed(_persisted(HandoffStatus.PENDING)) as ctx:
        get_response = client.get(f"{URL}/{HANDOFF_ID}", headers=_auth())
        patch_response = client.patch(
            f"{URL}/{HANDOFF_ID}/status",
            json={"status": "resolved"},
            headers=_auth(),
        )

    assert get_response.status_code == 200
    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "resolved"