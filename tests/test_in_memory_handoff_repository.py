"""Tests for the in-memory HandoffRepository (no network/env)."""

from datetime import date
from uuid import UUID

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import HandoffReason, HandoffRequest, PersistedHandoff
from app.repositories.handoff_repository import (
    HandoffRepository,
    HandoffRepositoryDuplicateError,
)
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository


def _state() -> ConversationState:
    return ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )


def _request() -> HandoffRequest:
    return HandoffRequest(
        customer_phone="+90555 111 2233",
        customer_name="Mehmet Cam",
        reason=HandoffReason.BOOKING_REVIEW,
        conversation_state=_state(),
    )


KEY = "c" * 64
OTHER_KEY = "d" * 64


def test_subclasses_handoff_repository():
    assert issubclass(InMemoryHandoffRepository, HandoffRepository)


def test_create_returns_persisted_handoff():
    persisted = InMemoryHandoffRepository().create(_request(), KEY)
    assert isinstance(persisted, PersistedHandoff)
    assert persisted.customer_phone == "+90555 111 2233"
    assert persisted.idempotency_key == KEY
    assert persisted.reason is HandoffReason.BOOKING_REVIEW


def test_create_assigns_unique_uuids():
    repo = InMemoryHandoffRepository()
    a = repo.create(_request(), KEY)
    b = repo.create(_request(), OTHER_KEY)
    assert a.id != b.id
    assert isinstance(a.id, UUID)


def test_get_after_create_roundtrips():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    fetched = repo.get(persisted.id)
    assert fetched == persisted
    assert fetched.conversation_state.tour == "Ephesus tour"


def test_get_missing_id_returns_none():
    assert InMemoryHandoffRepository().get(UUID("0" * 32)) is None


def test_get_by_idempotency_key_returns_existing():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    fetched = repo.get_by_idempotency_key(KEY)
    assert fetched == persisted


def test_get_by_idempotency_key_missing_returns_none():
    repo = InMemoryHandoffRepository()
    repo.create(_request(), KEY)
    assert repo.get_by_idempotency_key(OTHER_KEY) is None


def test_same_key_duplicate_create_raises():
    repo = InMemoryHandoffRepository()
    repo.create(_request(), KEY)
    with pytest.raises(HandoffRepositoryDuplicateError):
        repo.create(_request(), KEY)


def test_get_returns_deep_copy():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    persisted.conversation_state.tour = "MUTATED"  # type: ignore[assignment]
    fetched = repo.get(persisted.id)
    assert fetched.conversation_state.tour == "Ephesus tour"


def test_multiple_different_keys_coexist():
    repo = InMemoryHandoffRepository()
    a = repo.create(_request(), KEY)
    b = repo.create(_request(), OTHER_KEY)
    assert a.id != b.id
    assert repo.get_by_idempotency_key(KEY).id == a.id
    assert repo.get_by_idempotency_key(OTHER_KEY).id == b.id


def test_no_environment_dependency(monkeypatch):
    import os

    snapshot = dict(os.environ)
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak")
    monkeypatch.setenv("DATABASE_URL", "leak")
    after_setup = dict(os.environ)
    repo = InMemoryHandoffRepository()
    repo.create(_request(), KEY)
    assert dict(os.environ) == after_setup
    assert set(os.environ) - set(snapshot) == {
        "OPENROUTER_API_KEY",
        "DATABASE_URL",
    }


# --- Lifecycle: update_status ----------------------------------------------------


from app.models.handoff import HandoffStatus  # noqa: E402
from app.repositories.handoff_repository import HandoffNotFoundError  # noqa: E402


def test_update_status_pending_to_in_review():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    updated = repo.update_status(persisted.id, HandoffStatus.IN_REVIEW)
    assert updated.status is HandoffStatus.IN_REVIEW
    assert repo.get(persisted.id).status is HandoffStatus.IN_REVIEW


def test_update_status_in_review_to_resolved():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    repo.update_status(persisted.id, HandoffStatus.IN_REVIEW)
    updated = repo.update_status(persisted.id, HandoffStatus.RESOLVED)
    assert updated.status is HandoffStatus.RESOLVED


def test_update_status_pending_to_cancelled():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    updated = repo.update_status(persisted.id, HandoffStatus.CANCELLED)
    assert updated.status is HandoffStatus.CANCELLED


def test_update_status_missing_id_raises_not_found():
    repo = InMemoryHandoffRepository()
    with pytest.raises(HandoffNotFoundError):
        repo.update_status(UUID("0" * 32), HandoffStatus.RESOLVED)


def test_update_status_previous_object_unchanged_and_new_instance():
    repo = InMemoryHandoffRepository()
    before = repo.create(_request(), KEY)
    updated = repo.update_status(before.id, HandoffStatus.IN_REVIEW)

    assert before is not updated
    assert before.status is HandoffStatus.PENDING
    assert updated.status is HandoffStatus.IN_REVIEW


def test_update_status_preserves_snapshot_fields():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    updated = repo.update_status(persisted.id, HandoffStatus.IN_REVIEW)

    assert updated.id == persisted.id
    assert updated.idempotency_key == KEY
    assert updated.customer_phone == "+90555 111 2233"
    assert updated.customer_name == "Mehmet Cam"
    assert updated.reason is HandoffReason.BOOKING_REVIEW
    assert updated.conversation_state == persisted.conversation_state


def test_by_key_index_returns_updated_status_object():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    repo.update_status(persisted.id, HandoffStatus.IN_REVIEW)
    fetched = repo.get_by_idempotency_key(KEY)
    assert fetched.status is HandoffStatus.IN_REVIEW


def test_deep_copy_semantics_maintained_after_update():
    repo = InMemoryHandoffRepository()
    persisted = repo.create(_request(), KEY)
    updated = repo.update_status(persisted.id, HandoffStatus.IN_REVIEW)
    updated.conversation_state.tour = "MUTATED"  # type: ignore[assignment]

    stored = repo.get(persisted.id)
    assert stored.conversation_state.tour == "Ephesus tour"

