"""Phase 6 Step 4: message-route handoff integration tests (fakes only)."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.db.connection import DatabaseNotConfiguredError
from app.main import app
from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.models.extraction import ExtractedEntities, StructuredExtraction
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.repositories.provider import RepositoryConfigurationError
from app.services.ai.base import AIProvider, AIProviderError

client = TestClient(app)
URL = "/api/v1/messages/process"


def _ready_state(**overrides) -> ConversationState:
    base = dict(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    base.update(overrides)
    return ConversationState(**base)


class FakeProvider(AIProvider):
    def __init__(self, reply: str = "Sure!"):
        self.reply = reply

    async def generate_reply(
        self, message: str, conversation_context: str | None = None
    ) -> str:
        return self.reply

    async def extract_entities(self, message: str) -> StructuredExtraction:
        return StructuredExtraction(entities=ExtractedEntities())


class ExplodingProvider(AIProvider):
    async def generate_reply(
        self, message: str, conversation_context: str | None = None
    ) -> str:
        raise AIProviderError("provider failed")

    async def extract_entities(self, message: str) -> StructuredExtraction:
        raise AIProviderError("provider failed")


class RecordingConversationRepo:
    def __init__(self, state: ConversationState | None = None):
        self.state = state or ConversationState()
        self.saved: list[ConversationState] = []

    def get(self, customer_phone: str) -> ConversationState:
        EVENTS.append("conversation_get")
        return self.state.model_copy(deep=True)

    def save(self, customer_phone: str, state: ConversationState) -> None:
        EVENTS.append("conversation_save")
        self.saved.append(state)

    def clear(self) -> None:
        raise NotImplementedError


class RecordingHandoffRepo(InMemoryHandoffRepository):
    def __init__(self):
        super().__init__()
        self.lookups = 0

    def create(self, request, idempotency_key):  # type: ignore[override]
        EVENTS.append("handoff_create")
        return super().create(request, idempotency_key)

    def get_by_idempotency_key(self, idempotency_key):  # type: ignore[override]
        EVENTS.append("handoff_lookup")
        self.lookups += 1
        return super().get_by_idempotency_key(idempotency_key)


EVENTS: list[str] = []


def _setup(state=None, provider=None, handoff_repo=None):
    del EVENTS[:]
    conv_repo = RecordingConversationRepo(state)
    handoff_repo = handoff_repo or RecordingHandoffRepo()
    provider = provider or FakeProvider()

    factory_calls = {"count": 0}

    def fake_handoff_factory():
        factory_calls["count"] += 1
        return handoff_repo

    patches = [
        patch("app.routes.messages.get_ai_provider", return_value=provider),
        patch(
            "app.routes.messages.get_conversation_repository",
            return_value=conv_repo,
        ),
        patch(
            "app.routes.messages.get_handoff_repository",
            side_effect=fake_handoff_factory,
        ),
    ]
    # Default pipeline outcome so eligible requests reach handoff evaluation.
    patches.append(_pipeline_returning(state if state is not None else _ready_state()))
    return {
        "patches": patches,
        "provider": provider,
        "conv_repo": conv_repo,
        "handoff_repo": handoff_repo,
        "factory_calls": factory_calls,
    }


def _post(message="Please book the tour", name=None, phone="+905551112233"):
    payload = {"from": phone, "message": message}
    if name is not None:
        payload["name"] = name
    return client.post(URL, json=payload)


def _pipeline_returning(state):
    import app.services.conversation_pipeline_service as pipeline_module

    return patch.object(
        pipeline_module.ConversationPipelineService,
        "process_message",
            new=AsyncMock(return_value=state),
    )


# --- Core creation behavior ---------------------------------------------------


def test_ready_for_review_creates_booking_review_handoff():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 200
    stored = list(ctx["handoff_repo"]._store.values())
    assert len(stored) == 1
    from app.models.handoff import HandoffReason, HandoffStatus

    assert stored[0].reason is HandoffReason.BOOKING_REVIEW
    assert stored[0].status is HandoffStatus.PENDING


def test_handoff_uses_updated_state_not_stale_state():
    stale = _ready_state(tour="OLD tour")
    updated = _ready_state(tour="NEW tour")
    ctx = _setup(state=stale)
    pipeline_patch = _pipeline_returning(updated)

    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], pipeline_patch:
        response = _post()

    assert response.status_code == 200
    stored = list(ctx["handoff_repo"]._store.values())
    assert len(stored) == 1
    assert stored[0].conversation_state.tour == "NEW tour"


def test_conversation_save_occurs_before_handoff_create():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200

    assert EVENTS.index("conversation_save") < EVENTS.index("handoff_create")


def test_handoff_receives_normalized_phone():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        _post(phone="  +90555   111   2233 ")

    stored = list(ctx["handoff_repo"]._store.values())
    assert stored[0].customer_phone == "+90555 111 2233"


def test_handoff_receives_normalized_customer_name():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        _post(name="  Maria   Lopez  ")

    stored = list(ctx["handoff_repo"]._store.values())
    assert stored[0].customer_name == "Maria Lopez"


def test_blank_name_becomes_none_in_handoff():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        _post(name="   ")

    stored = list(ctx["handoff_repo"]._store.values())
    assert len(stored) == 1
    assert stored[0].customer_name is None


# --- Reason matrix through the route -----------------------------------------


def _assert_reason(state, expected_reason):
    ctx = _setup()
    pipeline_patch = _pipeline_returning(state)
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], pipeline_patch:
        response = _post()

    assert response.status_code == 200
    stored = list(ctx["handoff_repo"]._store.values())
    if expected_reason is None:
        assert stored == []
    else:
        assert len(stored) == 1
        from app.models.handoff import HandoffReason

        assert stored[0].reason is expected_reason


def test_collecting_details_creates_no_handoff():
    _assert_reason(
        ConversationState(
            intent=ConversationIntent.BOOKING_REQUEST,
            booking_stage=BookingStage.COLLECTING_DETAILS,
        ),
        None,
    )


def test_informational_state_creates_no_handoff():
    _assert_reason(ConversationState(intent=ConversationIntent.TOUR_INFORMATION), None)


def test_human_request_creates_correct_handoff():
    from app.models.handoff import HandoffReason

    _assert_reason(
        ConversationState(intent=ConversationIntent.HUMAN_REQUEST),
        HandoffReason.HUMAN_REQUEST,
    )


def test_complaint_creates_correct_handoff():
    from app.models.handoff import HandoffReason

    _assert_reason(ConversationState(intent=ConversationIntent.COMPLAINT), HandoffReason.COMPLAINT)


def test_cancellation_request_creates_correct_handoff():
    from app.models.handoff import HandoffReason

    _assert_reason(
        ConversationState(intent=ConversationIntent.CANCELLATION_REQUEST),
        HandoffReason.CANCELLATION_REQUEST,
    )


def test_existing_booking_requiring_human_creates_correct_handoff():
    from app.models.handoff import HandoffReason

    _assert_reason(
        ConversationState(intent=ConversationIntent.EXISTING_BOOKING, needs_human=True),
        HandoffReason.EXISTING_BOOKING,
    )


def test_generic_requires_human_creates_safety_escalation():
    from app.models.handoff import HandoffReason

    _assert_reason(
        ConversationState(intent=ConversationIntent.GENERAL_QUESTION, needs_human=True),
        HandoffReason.SAFETY_ESCALATION,
    )


# --- Idempotency through the route --------------------------------------------


def test_repeated_same_logical_state_does_not_duplicate():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200
        assert _post().status_code == 200

    assert len(ctx["handoff_repo"]._store) == 1


def test_meaningful_booking_change_creates_new_handoff():
    ctx = _setup()
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        _pipeline_returning(_ready_state()),
    ):
        assert _post().status_code == 200
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        _pipeline_returning(_ready_state(tour="Pamukkale")),
    ):
        assert _post().status_code == 200

    assert len(ctx["handoff_repo"]._store) == 2


def test_cosmetic_optional_field_change_does_not_duplicate():
    ctx = _setup()
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        _pipeline_returning(_ready_state()),
    ):
        assert _post().status_code == 200
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        _pipeline_returning(
            _ready_state(children=1, hotel="Hotel A", cruise_ship="Ascent")
        ),
    ):
        assert _post().status_code == 200

    assert len(ctx["handoff_repo"]._store) == 1


# --- API response contract ----------------------------------------------------


def test_handoff_details_absent_from_http_response():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    body = response.json()
    assert body == {
        "success": True,
        "data": {
            "customer_phone": "+905551112233",
            "reply": "Sure!",
        },
    }
    text = response.text.lower()
    for forbidden in (
        "handoff_id",
        "handoff_reason",
        "handoff_status",
        "idempotency_key",
        "booking_stage",
        "conversation_state",
    ):
                assert forbidden not in text


# --- Error paths ---------------------------------------------------------------


def test_handoff_configuration_error_returns_safe_500():
    ctx = _setup()
    ctx["patches"][2] = patch(
        "app.routes.messages.get_handoff_repository",
        side_effect=RepositoryConfigurationError("postgres"),
    )
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Human review service is not configured correctly."
    }
    assert "postgres" not in response.text.lower()


def test_handoff_database_not_configured_returns_safe_503():
    repo = InMemoryHandoffRepository()

    def raise_no_db(key):
        raise DatabaseNotConfiguredError("postgresql://secret")

    repo.get_by_idempotency_key = raise_no_db  # type: ignore[method-assign]
    ctx = _setup(handoff_repo=repo)
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 503
    assert response.json() == {"detail": "Human review service is unavailable."}
    assert "postgresql" not in response.text.lower()


def test_conversation_save_failure_skips_handoff():
    conv_repo = RecordingConversationRepo(_ready_state())

    def failing_save(phone, state):
        EVENTS.append("conversation_save")
        raise DatabaseNotConfiguredError("no db")

    conv_repo.save = failing_save  # type: ignore[method-assign]
    ctx = _setup()
    ctx["patches"][1] = patch(
        "app.routes.messages.get_conversation_repository", return_value=conv_repo
    )
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 503
    assert "handoff_create" not in EVENTS
    assert ctx["factory_calls"]["count"] == 0


def test_extraction_failure_skips_handoff():
    ctx = _setup(provider=ExplodingProvider())
    ctx["patches"][3] = _pipeline_returning_error()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 502
    assert ctx["factory_calls"]["count"] == 0
    assert "handoff_create" not in EVENTS


def test_invalid_request_422_skips_handoff_factory():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = client.post(URL, json={"from": "+905551112233", "message": "   "})

    assert response.status_code == 422
    assert ctx["factory_calls"]["count"] == 0


def _pipeline_returning_error():
    import app.services.conversation_pipeline_service as pipeline_module

    return patch.object(
        pipeline_module.ConversationPipelineService,
        "process_message",
        new=AsyncMock(side_effect=AIProviderError("extraction failed")),
    )


def _failing_repo(error: Exception):
    class FailingRepo(InMemoryHandoffRepository):
        def create(self, request, idempotency_key):  # type: ignore[override]
            raise error

    return FailingRepo()


def test_handoff_persistence_failure_returns_safe_503():
    ctx = _setup(
        handoff_repo=_failing_repo(
            RuntimeError("duplicate key value violates unique constraint")
        )
    )
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 503
    assert response.json() == {"detail": "Human review service is unavailable."}


def test_raw_db_error_and_url_not_leaked():
    leaked_url = "postgresql://user:hunter2@neon.example/db"
    ctx = _setup(
        handoff_repo=_failing_repo(RuntimeError(f"psycopg boom {leaked_url}"))
    )
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 503
    assert leaked_url not in response.text
    assert "hunter2" not in response.text


def test_idempotency_key_not_leaked_on_failure():
    from app.models.handoff import (
        HandoffReason,
        build_handoff_idempotency_key,
    )

    key = build_handoff_idempotency_key(
        "+905551112233", _ready_state(), HandoffReason.BOOKING_REVIEW
    )
    ctx = _setup(handoff_repo=_failing_repo(RuntimeError(f"constraint failed {key}")))
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 503
    assert key not in response.text


def test_conversation_state_remains_saved_after_handoff_failure():
    ctx = _setup(handoff_repo=_failing_repo(RuntimeError("boom")))
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        _post()

    assert len(ctx["conv_repo"].saved) == 1


def test_prompt_injection_skips_handoff_factory_completely():
    from types import SimpleNamespace

    ctx = _setup()
    injection_patch = patch(
        "app.routes.messages.inspect_prompt",
        return_value=SimpleNamespace(is_safe=False),
    )
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        injection_patch,
    ):
        response = _post()

    assert response.status_code == 200
    assert ctx["factory_calls"]["count"] == 0
    assert "handoff_lookup" not in EVENTS
    assert "handoff_create" not in EVENTS


# --- Reply/safety ordering -----------------------------------------------------


def test_reply_ai_error_after_handoff_keeps_502_and_handoff_persisted():
    class ReplyFailProvider(FakeProvider):
        async def generate_reply(self, message, conversation_context=None):
            raise AIProviderError("reply failed")

    ctx = _setup(provider=ReplyFailProvider())
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        response = _post()

    assert response.status_code == 502
    assert len(ctx["handoff_repo"]._store) == 1
    assert len(ctx["conv_repo"].saved) == 1


def test_output_block_returns_fallback_and_keeps_handoff():
    from app.services.safe_ai_service import SafeAIOutcome, SafeAIResult

    ctx = _setup()
    blocked_result = SafeAIResult(
        reply="I can help you with our tours.",
        outcome=SafeAIOutcome.OUTPUT_BLOCKED,
    )
    safeai_patch = patch(
        "app.routes.messages.SafeAIService.generate_reply",
        new=AsyncMock(return_value=blocked_result),
    )
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        ctx["patches"][3],
        safeai_patch,
    ):
        response = _post()

    assert response.status_code == 200
    assert response.json()["data"]["reply"] == "I can help you with our tours."
    assert len(ctx["handoff_repo"]._store) == 1

    # Repeated call after output block does not duplicate the handoff.
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        ctx["patches"][3],
        safeai_patch,
    ):
        assert _post().status_code == 200
    assert len(ctx["handoff_repo"]._store) == 1


# --- Factory reuse -------------------------------------------------------------


def test_handoff_factory_called_at_most_once_per_request():
    ctx = _setup()
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200

    assert ctx["factory_calls"]["count"] == 1


def test_no_direct_postgres_or_db_imports_in_route():
    import inspect

    import app.routes.messages as route_module

    source = inspect.getsource(route_module)
    assert "PostgresHandoffRepository(" not in source
    assert "psycopg" not in source
    assert "database_connection" not in source


# --- Lifecycle isolation (Step 5) ------------------------------------------------


def test_route_creates_pending_handoff_and_never_transitions_status():
    from app.models.handoff import HandoffStatus

    ctx = _setup()
    update_spy = MagicMock(wraps=ctx["handoff_repo"].update_status)
    ctx["handoff_repo"].update_status = update_spy  # type: ignore[method-assign]

    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200
        assert _post().status_code == 200

    stored = list(ctx["handoff_repo"]._store.values())
    assert len(stored) == 1
    assert stored[0].status is HandoffStatus.PENDING
    update_spy.assert_not_called()


def test_route_source_has_no_lifecycle_calls():
    import inspect

    import app.routes.messages as route_module

    source = inspect.getsource(route_module)
    assert "update_status" not in source
    assert "HandoffLifecycleService" not in source
    assert "transition(" not in source


def test_ai_components_do_not_import_lifecycle_service():
    import inspect

    import app.routes.messages as route_module
    import app.services.safe_ai_service as safe_ai_module

    try:
        import app.services.ai.openrouter as openrouter_module
        openrouter_src = inspect.getsource(openrouter_module)
    except Exception:  # pragma: no cover - module layout guard
        openrouter_src = ""

    for module_src in (inspect.getsource(route_module), inspect.getsource(safe_ai_module), openrouter_src):
        assert "handoff_lifecycle_service" not in module_src
        assert "HandoffLifecycleService" not in module_src


def test_ensure_handoff_does_not_reopen_nonpending_handoffs():
    """Route re-request must leave IN_REVIEW status untouched (Step 3 rule)."""
    from app.models.handoff import build_handoff_idempotency_key

    ctx = _setup()
    repo = ctx["handoff_repo"]

    # First request creates PENDING via the route.
    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200

    # Simulate a human picking it up.
    persisted = next(iter(repo._store.values()))
    real_key = build_handoff_idempotency_key(
        "+905551112233", _ready_state(), __import__(
            "app.models.handoff", fromlist=["HandoffReason"]
        ).HandoffReason.BOOKING_REVIEW,
    )
    seeded = type(persisted)(
        id=persisted.id,
        idempotency_key=real_key,
        customer_phone=persisted.customer_phone,
        customer_name=None,
        reason=persisted.reason,
        status=__import__(
            "app.models.handoff", fromlist=["HandoffStatus"]
        ).HandoffStatus.IN_REVIEW,
        conversation_state=_ready_state(),
    )
    repo._store[seeded.id] = seeded
    repo._by_key[real_key] = seeded.id

    with ctx["patches"][0], ctx["patches"][1], ctx["patches"][2], ctx["patches"][3]:
        assert _post().status_code == 200

    assert repo.get(seeded.id).status is (
        __import__("app.models.handoff", fromlist=["HandoffStatus"]).HandoffStatus.IN_REVIEW
    )


# --- Step 6: review read-model regression -----------------------------------------


def test_route_created_handoff_maps_to_review_and_response_stays_clean():
    from app.models.handoff import build_handoff_idempotency_key
    from app.models.handoff_review import build_handoff_review

    ctx = _setup()
    with (
        ctx["patches"][0],
        ctx["patches"][1],
        ctx["patches"][2],
        ctx["patches"][3],
    ):
        response = _post(name="Maria Lopez")

    assert response.status_code == 200
    stored = next(iter(ctx["handoff_repo"]._store.values()))
    review = build_handoff_review(stored)

    # Review reflects the route-created booking snapshot.
    from app.models.handoff import HandoffReason, HandoffStatus

    assert review.reason is HandoffReason.BOOKING_REVIEW
    assert review.status is HandoffStatus.PENDING
    assert review.customer_name == "Maria Lopez"
    assert review.tour == "Ephesus tour"
    assert review.travel_date == date(2026, 9, 10)
    assert review.adults == 2
    assert review.booking_stage is BookingStage.READY_FOR_REVIEW

    # HTTP response still contains only success/customer_phone/reply.
    body = response.json()
    assert set(body) == {"success", "data"}
    assert set(body["data"]) == {"customer_phone", "reply"}
    text = response.text.lower()
    for forbidden in ("review", "handoff", "idempotency", "booking_stage"):
        assert forbidden not in text






