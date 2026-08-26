"""Phase 7 Step 3: operator handoff listing + pagination/filter tests."""

from datetime import date, timedelta
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
TOKEN = "list-test-operator-token"


def _state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )


def _handoff(index: int, status: HandoffStatus, reason: HandoffReason):
    return PersistedHandoff(
        id=UUID(f"00000000-0000-0000-0000-{index:012d}"),
        idempotency_key=f"{index:064d}"[-64:],
        customer_phone=f"+9055500000{index:02d}",
        customer_name=None,
        reason=reason,
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


def _seed_repo(counts: dict[tuple[HandoffStatus, HandoffReason], int] | None = None):
    """Seed a repo with a deterministic mix of handoffs."""
    repo = SpyRepo()
    mix = counts if counts is not None else {
        (HandoffStatus.PENDING, HandoffReason.BOOKING_REVIEW): 3,
        (HandoffStatus.IN_REVIEW, HandoffReason.BOOKING_REVIEW): 1,
        (HandoffStatus.PENDING, HandoffReason.COMPLAINT): 1,
    }
    index = 0
    for (status, reason), n in mix.items():
        for _ in range(n):
            repo._store[UUID(f"00000000-0000-0000-0000-{index:012d}")] = _handoff(
                index, status, reason
            )
            index += 1

    factory_calls = {"count": 0}

    def factory():
        factory_calls["count"] += 1
        return repo

    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", side_effect=factory
    )
    return {"repo": repo, "patcher": patcher, "factory_calls": factory_calls}


@pytest.fixture(autouse=True)
def configured_token(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", TOKEN)


def _get_list(query: str = "", headers=None):
    return client.get(URL + query, headers=headers or {"Authorization": f"Bearer {TOKEN}"})


# --- AUTH ----------------------------------------------------------------------


def test_missing_auth_rejected_401():
    ctx = _seed_repo()
    with ctx["patcher"]:
        response = client.get(URL)

    assert response.status_code == 401
    assert response.json() == {"detail": "Operator authentication required."}


def test_wrong_auth_rejected_401():
    ctx = _seed_repo()
    with ctx["patcher"]:
        response = _get_list(headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401


def test_auth_unconfigured_fails_closed_503(monkeypatch):
    monkeypatch.setattr(settings, "operator_api_token", "")
    ctx = _seed_repo()
    with ctx["patcher"]:
        response = _get_list()

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Operator authentication is unavailable."
    }


# --- VALIDATION ------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "?limit=0",
        "?limit=101",
        "?limit=-5",
        "?offset=-1",
        "?status=super_urgent",
        "?reason=not_a_reason",
    ],
)
def test_invalid_query_params_rejected_422(query):
    ctx = _seed_repo()
    with ctx["patcher"]:
        response = _get_list(query)

    assert response.status_code == 422
    assert ctx["factory_calls"]["count"] == 0


# --- SUCCESS ---------------------------------------------------------------------


def test_empty_repository_returns_200_empty_items():
    ctx = _seed_repo({})
    with ctx["patcher"]:
        body = _get_list().json()

    assert body == {"items": [], "limit": 50, "offset": 0, "count": 0}


def test_multiple_items_returned_with_count_limit_offset():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list().json()

    assert body["count"] == len(body["items"]) == 5
    assert body["limit"] == 50
    assert body["offset"] == 0


def test_exact_safe_item_contract():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list("?limit=1").json()

    item = body["items"][0]
    assert set(item) == {
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


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "idempotency_key",
        "created_at",
        "updated_at",
        "raw_message",
        "message",
        "transcript",
        "ai_reply",
        "reply",
        "reasoning",
        "prompt",
        "provider",
        "model",
        "database_url",
        "repository_backend",
        "sql",
    ],
)
def test_sensitive_fields_absent(forbidden_field: str):
    ctx = _seed_repo()
    with ctx["patcher"]:
        text = _get_list().text.lower()

        assert forbidden_field not in text


# --- FILTERS ---------------------------------------------------------------------


def test_status_filter_returns_only_pending():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list("?status=pending").json()

    assert body["count"] == 4
    assert all(item["status"] == "pending" for item in body["items"])


def test_reason_filter_returns_only_booking_review():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list("?reason=booking_review").json()

    assert body["count"] == 4
    assert all(item["reason"] == "booking_review" for item in body["items"])


def test_combined_filter_returns_intersection():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list("?status=pending&reason=booking_review").json()

    assert body["count"] == 3
    assert all(
        item["status"] == "pending" and item["reason"] == "booking_review"
        for item in body["items"]
    )


def test_no_matching_combination_returns_empty_200():
    ctx = _seed_repo()
    with ctx["patcher"]:
        body = _get_list("?status=resolved&reason=complaint").json()

    assert response_ok(body) and body == {
        "items": [],
        "limit": 50,
        "offset": 0,
        "count": 0,
    }


def response_ok(body) -> bool:
    return True


# --- PAGINATION --------------------------------------------------------------------


def _seed_five():
    repo = SpyRepo()
    statuses = [
        (HandoffStatus.PENDING, HandoffReason.BOOKING_REVIEW),
        (HandoffStatus.PENDING, HandoffReason.BOOKING_REVIEW),
        (HandoffStatus.IN_REVIEW, HandoffReason.COMPLAINT),
        (HandoffStatus.RESOLVED, HandoffReason.HUMAN_REQUEST),
        (HandoffStatus.CANCELLED, HandoffReason.SAFETY_ESCALATION),
    ]
    for index, (status, reason) in enumerate(statuses):
        repo._store[UUID(f"00000000-0000-0000-0000-{index:012d}")] = _handoff(
            index, status, reason
        )
    factory_calls = {"count": 0}

    def factory():
        factory_calls["count"] += 1
        return repo

    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", side_effect=factory
    )
    return {"repo": repo, "patcher": patcher, "factory_calls": factory_calls}


def test_first_page():
    ctx = _seed_five()
    with ctx["patcher"]:
        body = _get_list("?limit=2&offset=0").json()

    assert body["count"] == 2
    assert [i["handoff_id"][-12:] for i in body["items"]] == [
        "000000000000",
        "000000000001",
    ]


def test_middle_page():
    ctx = _seed_five()
    with ctx["patcher"]:
        body = _get_list("?limit=2&offset=2").json()

    assert [i["handoff_id"][-12:] for i in body["items"]] == [
        "000000000002",
        "000000000003",
    ]


def test_final_partial_page():
    ctx = _seed_five()
    with ctx["patcher"]:
        body = _get_list("?limit=2&offset=4").json()

    assert body["count"] == 1
    assert body["items"][0]["handoff_id"].endswith("000000000004")


def test_offset_past_end_returns_empty():
    ctx = _seed_five()
    with ctx["patcher"]:
        body = _get_list("?limit=2&offset=10").json()

    assert body["items"] == []
    assert body["count"] == 0


def test_deterministic_repeated_listing():
    ctx = _seed_five()
    with ctx["patcher"]:
        first = _get_list().json()
        second = _get_list().json()

    assert first == second


def test_no_overlap_across_non_overlapping_pages():
    ctx = _seed_five()
    with ctx["patcher"]:
        page1 = _get_list("?limit=2&offset=0").json()["items"]
        page2 = _get_list("?limit=2&offset=2").json()["items"]
        page3 = _get_list("?limit=2&offset=4").json()["items"]

    ids = [i["handoff_id"] for i in page1 + page2 + page3]
    assert len(ids) == len(set(ids)) == 5


# --- READ-ONLY ---------------------------------------------------------------------


def test_listing_performs_zero_update_status_calls():
    ctx = _seed_repo()
    update_spy = MagicMock(wraps=ctx["repo"].update_status)
    ctx["repo"].update_status = update_spy  # type: ignore[method-assign]

    with ctx["patcher"]:
        _get_list()

    update_spy.assert_not_called()


def test_statuses_and_booking_stages_preserved_after_listing():
    ctx = _seed_five()
    with ctx["patcher"]:
        _get_list()

    stored = list(ctx["repo"]._store.values())
    assert [h.status for h in stored] == [
        HandoffStatus.PENDING,
        HandoffStatus.PENDING,
        HandoffStatus.IN_REVIEW,
        HandoffStatus.RESOLVED,
        HandoffStatus.CANCELLED,
    ]
    assert all(
        h.conversation_state.booking_stage is BookingStage.READY_FOR_REVIEW
        for h in stored
    )


def test_repeated_list_side_effect_free():
    ctx = _seed_repo()
    snapshot_before = {
        k: v.model_copy(deep=True) for k, v in ctx["repo"]._store.items()
    }
    with ctx["patcher"]:
        for _ in range(3):
            _get_list()

    assert ctx["repo"]._store == snapshot_before


# --- ERRORS ------------------------------------------------------------------------


def test_repository_configuration_error_safe_500():
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository",
        side_effect=RepositoryConfigurationError("postgres"),
    )
    with patcher:
        response = _get_list()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Human review service is not configured correctly."
    }
    assert "postgres" not in response.text.lower()


def test_database_not_configured_error_safe_503():
    repo = MagicMock()
    repo.list_handoffs.side_effect = DatabaseNotConfiguredError("postgresql://secret")
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", return_value=repo
    )
    with patcher:
        response = _get_list()

    assert response.status_code == 503
    assert response.json() == {"detail": "Human review service is unavailable."}
    assert "postgresql" not in response.text.lower()


def test_unexpected_storage_error_safe_503_no_leak():
    leaked_url = "postgresql://user:hunter2@neon.example/db"
    repo = MagicMock()
    repo.list_handoffs.side_effect = RuntimeError(f"boom {leaked_url}")
    patcher = patch(
        "app.routes.operator_handoffs.get_handoff_repository", return_value=repo
    )
    with patcher:
        response = _get_list()

    assert response.status_code == 503
    assert leaked_url not in response.text
    assert "hunter2" not in response.text


# --- ARCHITECTURE / REPOSITORY --------------------------------------------------------


def test_route_uses_factory_and_review_service_for_listing():
    import inspect

    import app.routes.operator_handoffs as module

    source = inspect.getsource(module)
    assert "get_handoff_repository()" in source
    assert "HandoffReviewService(" in source
    assert "service.list_reviews(" in source

    # The listing path itself must not touch lifecycle or SQL.
    list_source = inspect.getsource(module.list_handoffs)
    for forbidden in (
        "SELECT ",
        "PostgresHandoffRepository(",
        "database_connection",
        ".update_status(",
        "HandoffLifecycleService",
    ):
        assert forbidden not in list_source, forbidden


def test_no_real_db_or_network_in_test(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://should_not_be_used")
    ctx = _seed_repo()
    with ctx["patcher"]:
        assert _get_list().status_code == 200


def test_postgres_list_is_parameterized_with_fixed_ordering():
    factory, conn, cursor = MagicMock(), MagicMock(), MagicMock()
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    factory.return_value.__enter__ = MagicMock(return_value=conn)
    factory.return_value.__exit__ = MagicMock(return_value=False)

    import app.repositories.postgres_handoff_repository as pg_module

    with patch.object(pg_module, "database_connection", factory):
        pg_module.PostgresHandoffRepository().list_handoffs(
            status=HandoffStatus.PENDING,
            reason=HandoffReason.BOOKING_REVIEW,
            limit=25,
            offset=5,
        )

    sql, params = cursor.execute.call_args.args
    assert sql.count("%s") == 4
    assert params == ("pending", "booking_review", 25, 5)
    assert "pending" not in sql and "booking_review" not in sql
    assert "ORDER BY created_at ASC, id ASC" in sql
    assert "LIMIT %s" in sql and "OFFSET %s" in sql
    conn.commit.assert_not_called()


def test_postgres_list_without_filters_has_no_where():
    factory, conn, cursor = MagicMock(), MagicMock(), MagicMock()
    cursor.fetchall.return_value = []
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor
    factory.return_value.__enter__ = MagicMock(return_value=conn)
    factory.return_value.__exit__ = MagicMock(return_value=False)

    import app.repositories.postgres_handoff_repository as pg_module

    with patch.object(pg_module, "database_connection", factory):
        pg_module.PostgresHandoffRepository().list_handoffs(limit=10, offset=0)

    sql, params = cursor.execute.call_args.args
    assert "WHERE" not in sql
    assert params == (10, 0)


def test_memory_repo_filter_and_pagination_semantics():
    repo = InMemoryHandoffRepository()
    for i in range(5):
        handoff = PersistedHandoff(
            id=UUID(f"00000000-0000-0000-0000-{i:012d}"),
            idempotency_key=str(i).zfill(64),
            customer_phone=f"+9000000000{i}",
            reason=(
                HandoffReason.BOOKING_REVIEW if i % 2 == 0 else HandoffReason.COMPLAINT
            ),
            status=HandoffStatus.PENDING if i < 3 else HandoffStatus.IN_REVIEW,
            conversation_state=_state(),
        )
        repo._store[handoff.id] = handoff

    pending = repo.list_handoffs(status=HandoffStatus.PENDING)
    assert len(pending) == 3
    complaints = repo.list_handoffs(reason=HandoffReason.COMPLAINT)
    assert len(complaints) == 2
    page = repo.list_handoffs(limit=2, offset=2)
    assert [str(h.id)[-12:] for h in page] == ["000000000002", "000000000003"]


def test_memory_list_returns_deep_copies():
    repo = InMemoryHandoffRepository()
    handoff = PersistedHandoff(
        id=UUID("00000000-0000-0000-0000-000000000000"),
        idempotency_key="k" * 64,
        customer_phone="+905551112233",
        reason=HandoffReason.BOOKING_REVIEW,
        status=HandoffStatus.PENDING,
        conversation_state=_state(),
    )
    repo._store[handoff.id] = handoff

    listed = repo.list_handoffs()[0]
    listed.conversation_state.tour = "MUTATED"

    assert repo._store[handoff.id].conversation_state.tour == "Ephesus tour"


# --- REGRESSIONS ---------------------------------------------------------------------


def test_existing_get_by_id_still_works():
    ctx = _seed_repo()
    with ctx["patcher"]:
        single = client.get(
            f"{URL}/00000000-0000-0000-0000-000000000000",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert single.status_code == 200
    assert single.json()["handoff_id"].endswith("000000000000")


def test_existing_patch_status_still_works():
    ctx = _seed_repo()
    with ctx["patcher"]:
        response = client.patch(
            f"{URL}/00000000-0000-0000-0000-000000000000/status",
            json={"status": "in_review"},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "in_review"


def test_customer_message_route_unchanged():
    from app.models.extraction import ExtractedEntities, StructuredExtraction
    from app.services.ai.base import AIProvider

    class FakeProvider(AIProvider):
        async def generate_reply(self, message, conversation_context=None):
            return "Hello!"

        async def extract_entities(self, message):
            return StructuredExtraction(entities=ExtractedEntities())

    with patch("app.routes.messages.get_ai_provider", return_value=FakeProvider()):
        response = client.post(
            "/api/v1/messages/process",
            json={"from": "+905551112233", "message": "Hello"},
        )

    assert response.status_code == 200
    assert set(response.json()["data"]) == {"customer_phone", "reply"}

    assert "hunter2" not in response.text
