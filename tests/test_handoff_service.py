"""Tests for HandoffService and the idempotency key (Phase 6 Step 3)."""

from datetime import date
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from app.models.conversation import (
    BookingStage,
    ConversationIntent,
    ConversationState,
)
from app.models.handoff import (
    HandoffReason,
    HandoffStatus,
    PersistedHandoff,
    build_handoff_idempotency_key,
)
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.repositories.handoff_repository import HandoffRepository
from app.services.handoff_service import HandoffService


def _state(**overrides) -> ConversationState:
    base = dict(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus tour",
        travel_date=date(2026, 9, 10),
        adults=2,
        booking_stage=BookingStage.READY_FOR_REVIEW,
    )
    base.update(overrides)
    return ConversationState(**base)


PHONE = "+90555 111 2233"
KEY = "e" * 64


def _persisted(
    status: HandoffStatus = HandoffStatus.PENDING,
    state: ConversationState | None = None,
    reason: HandoffReason = HandoffReason.BOOKING_REVIEW,
) -> PersistedHandoff:
    return PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key=KEY,
        customer_phone="+90555 111 2233",
        reason=reason,
        status=status,
        conversation_state=state or _state(),
    )


# --- Idempotency key --------------------------------------------------------


def test_key_is_deterministic():
    a = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    assert a == b


def test_normalized_phone_gives_same_key():
    a = build_handoff_idempotency_key("+90555 111 2233", _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key("  +90555   111 2233 ", _state(), HandoffReason.BOOKING_REVIEW)
    assert a == b


def test_different_reason_different_key():
    a = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.COMPLAINT)
    assert a != b


def test_different_tour_different_key():
    a = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key(
        PHONE, _state(tour="Pamukkale"), HandoffReason.BOOKING_REVIEW
    )
    assert a != b


def test_different_travel_date_different_key():
    a = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key(
        PHONE, _state(travel_date=date(2026, 9, 11)), HandoffReason.BOOKING_REVIEW
    )
    assert a != b


def test_different_adults_different_key():
    a = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    b = build_handoff_idempotency_key(
        PHONE, _state(adults=4), HandoffReason.BOOKING_REVIEW
    )
    assert a != b


@pytest.mark.parametrize(
    "override",
    [
        {"children": 1},
        {"hotel": "Hotel A"},
        {"cruise_ship": "Celebrity Ascent"},
        {"pickup_location": "Kusadasi Port"},
        {"preferred_language": "Turkish"},
    ],
)
def test_optional_fields_irrelevant_to_key(override):
    base = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    changed = build_handoff_idempotency_key(
        PHONE, _state(**override), HandoffReason.BOOKING_REVIEW
    )
    assert base == changed


def test_output_is_64_char_lowercase_hex():
    key = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    assert len(key) == 64
    assert key == key.lower()
    int(key, 16)  # valid hex


def test_raw_phone_not_visible_in_key():
    key = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    assert "90555" not in key
    assert "2233" not in key


def test_no_network_or_env_dependency(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak")
    monkeypatch.setenv("DATABASE_URL", "leak")
    key = build_handoff_idempotency_key(PHONE, _state(), HandoffReason.BOOKING_REVIEW)
    assert len(key) == 64


# --- Service -----------------------------------------------------------------


def _service_with_repo(repo=None):
    repo = repo or InMemoryHandoffRepository()
    return HandoffService(repo), repo


def test_reason_none_returns_none_and_repository_untouched():
    service, repo = _service_with_repo()
    spy = MagicMock(wraps=repo)
    service._repository = spy

    result = service.ensure_handoff(
        PHONE,
        _state(
            intent=ConversationIntent.TOUR_INFORMATION,
            booking_stage=BookingStage.NONE,
        ),
    )

    assert result is None
    spy.get_by_idempotency_key.assert_not_called()
    spy.create.assert_not_called()


def test_ready_for_review_creates_booking_review():
    service, repo = _service_with_repo()
    persisted = service.ensure_handoff(PHONE, _state())
    assert persisted is not None
    assert persisted.reason is HandoffReason.BOOKING_REVIEW
    assert persisted.status is HandoffStatus.PENDING


def test_second_same_call_returns_existing():
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state())
    second = service.ensure_handoff(PHONE, _state())
    assert second is not None and first is not None
    assert second.id == first.id


def test_create_count_is_one_for_duplicate():
    service, repo = _service_with_repo()
    service.ensure_handoff(PHONE, _state())
    service.ensure_handoff(PHONE, _state())
    service.ensure_handoff(PHONE, _state())
    assert len(repo._store) == 1


@pytest.mark.parametrize(
    "status",
    [HandoffStatus.IN_REVIEW, HandoffStatus.RESOLVED, HandoffStatus.CANCELLED],
)
def test_existing_status_is_preserved(status):
    repo = InMemoryHandoffRepository()
    from app.models.handoff import build_handoff_idempotency_key

    real_key = build_handoff_idempotency_key(
        PHONE, _state(), HandoffReason.BOOKING_REVIEW
    )
    seeded = PersistedHandoff(
        id=UUID("12345678123456781234567812345678"),
        idempotency_key=real_key,
        customer_phone="+90555 111 2233",
        reason=HandoffReason.BOOKING_REVIEW,
        status=status,
        conversation_state=_state(),
    )
    repo._store[seeded.id] = seeded
    repo._by_key[real_key] = seeded.id

    service = HandoffService(repo)
    result = service.ensure_handoff(PHONE, _state())

    assert result is not None
    assert result.status is status
    assert result.id == seeded.id


def test_meaningful_change_tour_creates_new_handoff():
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state())
    second = service.ensure_handoff(PHONE, _state(tour="Pamukkale"))
    assert first is not None and second is not None
    assert first.id != second.id
    assert len(repo._store) == 2


def test_meaningful_change_travel_date_creates_new_handoff():
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state())
    second = service.ensure_handoff(PHONE, _state(travel_date=date(2026, 9, 11)))
    assert first.id != second.id
    assert len(repo._store) == 2


def test_meaningful_change_adults_creates_new_handoff():
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state())
    second = service.ensure_handoff(PHONE, _state(adults=4))
    assert first.id != second.id
    assert len(repo._store) == 2


@pytest.mark.parametrize(
    "override",
    [
        {"children": 1},
        {"hotel": "Hotel A"},
        {"cruise_ship": "Celebrity Ascent"},
    ],
)
def test_optional_field_changes_do_not_duplicate(override):
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state())
    second = service.ensure_handoff(PHONE, _state(**override))
    assert first is not None and second is not None
    assert first.id == second.id
    assert len(repo._store) == 1


def test_customer_name_change_does_not_duplicate():
    service, repo = _service_with_repo()
    first = service.ensure_handoff(PHONE, _state(), customer_name="Maria")
    second = service.ensure_handoff(
        PHONE, _state(), customer_name="Maria Lopez"
    )
    assert first is not None and second is not None
    assert first.id == second.id
    assert len(repo._store) == 1


@pytest.mark.parametrize(
    ("intent", "expected_reason"),
    [
        (ConversationIntent.CANCELLATION_REQUEST, HandoffReason.CANCELLATION_REQUEST),
        (ConversationIntent.COMPLAINT, HandoffReason.COMPLAINT),
        (ConversationIntent.HUMAN_REQUEST, HandoffReason.HUMAN_REQUEST),
    ],
)
def test_non_booking_reasons_create_correct_handoffs(intent, expected_reason):
    service, _ = _service_with_repo()
    state = ConversationState(intent=intent)
    persisted = service.ensure_handoff(PHONE, state)
    assert persisted is not None
    assert persisted.reason is expected_reason


def test_generic_safety_escalation_works():
    service, _ = _service_with_repo()
    state = ConversationState(
        intent=ConversationIntent.GENERAL_QUESTION, needs_human=True
    )
    persisted = service.ensure_handoff(PHONE, state)
    assert persisted is not None
    assert persisted.reason is HandoffReason.SAFETY_ESCALATION


def test_state_snapshot_remains_independent():
    service, _ = _service_with_repo()
    source = _state()
    persisted = service.ensure_handoff(PHONE, source)
    source.tour = "MUTATED"
    source.adults = 99
    assert persisted.conversation_state.tour == "Ephesus tour"
    assert persisted.conversation_state.adults == 2


def test_phone_and_name_normalization_preserved():
    service, _ = _service_with_repo()
    persisted = service.ensure_handoff(
        "  +90555   111   2233 ", _state(), customer_name="  Mehmet   Cam "
    )
    assert persisted.customer_phone == "+90555 111 2233"
    assert persisted.customer_name == "Mehmet Cam"


def test_repository_error_propagates():
    failing_repo = MagicMock(spec=HandoffRepository)
    failing_repo.get_by_idempotency_key.return_value = None
    failing_repo.create.side_effect = RuntimeError("db exploded")
    service = HandoffService(failing_repo)

    with pytest.raises(RuntimeError, match="db exploded"):
        service.ensure_handoff(PHONE, _state())


def test_no_ai_or_network_dependency(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "leak")
    monkeypatch.setenv("DATABASE_URL", "leak")
    service, _ = _service_with_repo(InMemoryHandoffRepository())
    persisted = service.ensure_handoff(PHONE, _state())
    assert persisted is not None

