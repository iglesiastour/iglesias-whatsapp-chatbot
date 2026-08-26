"""Phase 6 Step 5: handoff lifecycle transition matrix and service tests."""

from datetime import date
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
)
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.services.handoff_lifecycle_service import (
    HandoffLifecycleService,
    InvalidHandoffTransitionError,
    validate_handoff_transition,
)

FIXED_ID = UUID("12345678123456781234567812345678")


def _state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )


def _seed(status: HandoffStatus = HandoffStatus.PENDING):
    repo = InMemoryHandoffRepository()
    persisted = PersistedHandoff(
        id=FIXED_ID,
        idempotency_key="k" * 64,
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        status=status,
        conversation_state=_state(),
    )
    repo._store[FIXED_ID] = persisted
    repo._by_key[persisted.idempotency_key] = FIXED_ID
    return repo, persisted


# --- Transition matrix ---------------------------------------------------------

ALLOWED = [
    (HandoffStatus.PENDING, HandoffStatus.PENDING),
    (HandoffStatus.PENDING, HandoffStatus.IN_REVIEW),
    (HandoffStatus.PENDING, HandoffStatus.RESOLVED),
    (HandoffStatus.PENDING, HandoffStatus.CANCELLED),
    (HandoffStatus.IN_REVIEW, HandoffStatus.IN_REVIEW),
    (HandoffStatus.IN_REVIEW, HandoffStatus.RESOLVED),
    (HandoffStatus.IN_REVIEW, HandoffStatus.CANCELLED),
    (HandoffStatus.RESOLVED, HandoffStatus.RESOLVED),
    (HandoffStatus.CANCELLED, HandoffStatus.CANCELLED),
]

INVALID = [
    (HandoffStatus.IN_REVIEW, HandoffStatus.PENDING),
    (HandoffStatus.RESOLVED, HandoffStatus.PENDING),
    (HandoffStatus.RESOLVED, HandoffStatus.IN_REVIEW),
    (HandoffStatus.RESOLVED, HandoffStatus.CANCELLED),
    (HandoffStatus.CANCELLED, HandoffStatus.PENDING),
    (HandoffStatus.CANCELLED, HandoffStatus.IN_REVIEW),
    (HandoffStatus.CANCELLED, HandoffStatus.RESOLVED),
]


@pytest.mark.parametrize(("current", "target"), ALLOWED)
def test_allowed_transitions_validate(current, target):
    assert validate_handoff_transition(current, target) is None


@pytest.mark.parametrize(("current", "target"), INVALID)
def test_invalid_transitions_raise(current, target):
    with pytest.raises(InvalidHandoffTransitionError):
        validate_handoff_transition(current, target)


def test_validation_is_deterministic():
    for _ in range(3):
        with pytest.raises(InvalidHandoffTransitionError):
            validate_handoff_transition(HandoffStatus.RESOLVED, HandoffStatus.PENDING)
    for _ in range(3):
        assert (
            validate_handoff_transition(HandoffStatus.PENDING, HandoffStatus.IN_REVIEW)
            is None
        )


def test_error_messages_contain_no_sensitive_data():
    transition_error = str(InvalidHandoffTransitionError())
    not_found_error = str(HandoffNotFoundError())
    for text in (transition_error, not_found_error):
        assert "12345678" not in text
        assert "90555" not in text
        assert "k" * 8 not in text
        assert "postgresql" not in text.lower()
    assert transition_error == "Invalid handoff status transition."
    assert not_found_error == "Handoff not found."


# --- Service behavior ------------------------------------------------------------


def test_missing_handoff_raises_not_found():
    service = HandoffLifecycleService(InMemoryHandoffRepository())
    with pytest.raises(HandoffNotFoundError):
        service.transition(uuid4(), HandoffStatus.IN_REVIEW)


def test_allowed_transition_updates_repository():
    repo, _ = _seed(HandoffStatus.PENDING)
    service = HandoffLifecycleService(repo)
    updated = service.transition(FIXED_ID, HandoffStatus.IN_REVIEW)
    assert updated.status is HandoffStatus.IN_REVIEW
    assert repo.get(FIXED_ID).status is HandoffStatus.IN_REVIEW


def test_same_status_transition_skips_update_call():
    repo, _ = _seed(HandoffStatus.PENDING)
    update_spy = MagicMock(wraps=repo.update_status)
    repo.update_status = update_spy  # type: ignore[method-assign]
    service = HandoffLifecycleService(repo)

    result = service.transition(FIXED_ID, HandoffStatus.PENDING)

    update_spy.assert_not_called()
    assert result.status is HandoffStatus.PENDING


def test_invalid_transition_skips_update_call():
    repo, _ = _seed(HandoffStatus.RESOLVED)
    update_spy = MagicMock(wraps=repo.update_status)
    repo.update_status = update_spy  # type: ignore[method-assign]
    service = HandoffLifecycleService(repo)

    with pytest.raises(InvalidHandoffTransitionError):
        service.transition(FIXED_ID, HandoffStatus.PENDING)

    update_spy.assert_not_called()


def test_returned_object_has_target_status_and_previous_unchanged():
    repo, previous = _seed(HandoffStatus.PENDING)
    service = HandoffLifecycleService(repo)
    updated = service.transition(FIXED_ID, HandoffStatus.IN_REVIEW)

    assert updated.status is HandoffStatus.IN_REVIEW
    assert previous.status is HandoffStatus.PENDING
    assert updated is not previous


def test_identity_fields_preserved_after_transition():
    repo, previous = _seed(HandoffStatus.PENDING)
    service = HandoffLifecycleService(repo)
    updated = service.transition(FIXED_ID, HandoffStatus.RESOLVED)

    assert updated.id == previous.id
    assert updated.idempotency_key == previous.idempotency_key
    assert updated.reason is previous.reason
    assert updated.customer_phone == previous.customer_phone
    assert updated.customer_name == previous.customer_name
    assert updated.conversation_state == previous.conversation_state


# --- BookingStage independence ---------------------------------------------------


@pytest.mark.parametrize("target", [HandoffStatus.RESOLVED, HandoffStatus.CANCELLED])
def test_transition_does_not_mutate_booking_stage(target):
    repo, previous = _seed(HandoffStatus.PENDING)
    service = HandoffLifecycleService(repo)
    updated = service.transition(FIXED_ID, target)

    assert updated.conversation_state.booking_stage is BookingStage.READY_FOR_REVIEW
    stored = repo.get(FIXED_ID)
    assert stored.conversation_state.booking_stage is BookingStage.READY_FOR_REVIEW
    assert stored.status is target


def test_repository_error_propagates():
    failing = MagicMock(spec=HandoffRepository)
    failing.get.return_value = None
    service = HandoffLifecycleService(failing)
    with pytest.raises(HandoffNotFoundError):
        service.transition(FIXED_ID, HandoffStatus.RESOLVED)


def test_no_ai_or_network_dependency(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak")
    monkeypatch.setenv("DATABASE_URL", "leak")
    repo, _ = _seed()
    service = HandoffLifecycleService(repo)
    result = service.transition(FIXED_ID, HandoffStatus.CANCELLED)
    assert result.status is HandoffStatus.CANCELLED

