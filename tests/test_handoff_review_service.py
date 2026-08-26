"""Tests for HandoffReviewService (read-only, fakes only)."""

from datetime import date
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.db.connection import DatabaseNotConfiguredError
from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff
from app.models.handoff_review import HandoffReview, build_handoff_review
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
)
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.services.handoff_review_service import HandoffReviewService


def _state(**overrides) -> ConversationState:
    base = dict(
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
    base.update(overrides)
    return ConversationState(**base)


def _persisted(
    status: HandoffStatus = HandoffStatus.PENDING,
    state: ConversationState | None = None,
) -> PersistedHandoff:
    return PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key="k" * 64,
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        status=status,
        conversation_state=state or _state(),
    )


def _seed(status: HandoffStatus = HandoffStatus.PENDING, state=None):
    repo = InMemoryHandoffRepository()
    handoff = _persisted(status=status, state=state)
    # Mimic real repository semantics: a deep-copied snapshot is stored.
    repo._store[handoff.id] = handoff.model_copy(deep=True)
    repo._by_key[handoff.idempotency_key] = handoff.id
    return repo, handoff


def test_build_review_from_full_persisted_handoff():
    repo, handoff = _seed()
    review = HandoffReviewService(repo).get_review(handoff.id)

    assert isinstance(review, HandoffReview)
    assert review.handoff_id == handoff.id
    assert review.customer_phone == "+90555 111 2233"
    assert review.customer_name == "Mehmet Cam"
    assert review.reason is HandoffReason.BOOKING_REVIEW
    assert review.status is HandoffStatus.PENDING
    assert review.intent is ConversationIntent.BOOKING_REQUEST
    assert review.booking_stage is BookingStage.READY_FOR_REVIEW
    assert review.tour == "Ephesus tour"
    assert review.travel_date == date(2026, 9, 10)
    assert review.adults == 2
    assert review.children == 1
    assert review.cruise_ship == "Equinox"
    assert review.hotel == "Korumar"
    assert review.pickup_location == "Port"
    assert review.preferred_language == "English"
    assert review.needs_human is True


def test_missing_record_raises_not_found():
    service = HandoffReviewService(InMemoryHandoffRepository())
    with pytest.raises(HandoffNotFoundError):
        service.get_review(uuid4())


def test_get_review_calls_repository_get_exactly_once():
    repo, handoff = _seed()
    spy = MagicMock(wraps=repo.get)
    repo.get = spy  # type: ignore[method-assign]

    HandoffReviewService(repo).get_review(handoff.id)

    spy.assert_called_once()


def test_get_review_performs_zero_update_status_calls():
    repo, handoff = _seed()
    update_spy = MagicMock(wraps=repo.update_status)
    repo.update_status = update_spy  # type: ignore[method-assign]

    HandoffReviewService(repo).get_review(handoff.id)

    update_spy.assert_not_called()


@pytest.mark.parametrize("status", list(HandoffStatus))
def test_status_unchanged_after_read(status):
    repo, handoff = _seed(status=status)
    review = HandoffReviewService(repo).get_review(handoff.id)

    assert review.status is status
    assert repo.get(handoff.id).status is status


def test_resolved_handoff_with_ready_for_review_booking_stage():
    repo, _ = _seed(status=HandoffStatus.RESOLVED)
    review = HandoffReviewService(repo).get_review(_persisted().id)
    assert review.status is HandoffStatus.RESOLVED
    assert review.booking_stage is BookingStage.READY_FOR_REVIEW


def test_cancelled_handoff_with_confirmed_booking_stage_no_reconciliation():
    repo, _ = _seed(
        status=HandoffStatus.CANCELLED,
        state=_state(booking_stage=BookingStage.CONFIRMED),
    )
    review = HandoffReviewService(repo).get_review(_persisted().id)
    assert review.status is HandoffStatus.CANCELLED
    assert review.booking_stage is BookingStage.CONFIRMED


def test_snapshot_not_affected_by_later_live_state_change():
    live = _state()
    repo, handoff = _seed(state=live)
    service = HandoffReviewService(repo)
    review_before = service.get_review(handoff.id)

    # Live conversation state changes after the handoff was persisted.
    live.tour = "COMPLETELY DIFFERENT"
    live.adults = 42

    review_after = service.get_review(handoff.id)
    assert review_before == review_after
    assert review_after.tour == "Ephesus tour"
    assert review_after.adults == 2


def test_returned_review_is_immutable():
    repo, handoff = _seed()
    review = HandoffReviewService(repo).get_review(handoff.id)
    with pytest.raises(Exception):
        review.status = HandoffStatus.RESOLVED  # type: ignore[misc]


def test_deterministic_repeated_mapping():
    repo, handoff = _seed()
    service = HandoffReviewService(repo)
    assert service.get_review(handoff.id) == build_handoff_review(
        repo.get(handoff.id)
    )


def test_repository_errors_propagate_unchanged():
    failing = MagicMock(spec=HandoffRepository)
    failing.get.side_effect = DatabaseNotConfiguredError("no url")
    service = HandoffReviewService(failing)
    with pytest.raises(DatabaseNotConfiguredError):
        service.get_review(UUID("12345678123456781234567812345678"))


def test_no_ai_network_or_env_dependency(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak")
    monkeypatch.setenv("DATABASE_URL", "leak")
    repo, handoff = _seed()
    review = HandoffReviewService(repo).get_review(handoff.id)
    assert review.status is HandoffStatus.PENDING


def test_service_module_has_no_forbidden_imports():
    import inspect

    import app.services.handoff_review_service as module

    source = inspect.getsource(module)
    for forbidden in (
        "PostgresHandoffRepository",
        "database_connection",
        "psycopg",
        "OpenRouterProvider",
        "SafeAIService",
        "routes.messages",
        "update_status",
        "HandoffLifecycleService",
    ):
        assert forbidden not in source, forbidden

