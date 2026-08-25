"""Deterministic tests for the in-memory conversation store."""

import pytest

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.services.conversation_store import (
    InMemoryConversationStore,
    get_conversation_store,
)


@pytest.fixture()
def store() -> InMemoryConversationStore:
    store = InMemoryConversationStore()
    store.clear()
    return store


def test_missing_customer_returns_default_state(store) -> None:
    state = store.get("+905551112233")
    assert isinstance(state, ConversationState)
    assert state == ConversationState()


def test_get_returns_new_object_not_internal(store) -> None:
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    first = store.get("+905551112233")
    second = store.get("+905551112233")
    assert first is not second


def test_save_stores_state(store) -> None:
    state = ConversationState(tour="Ephesus", adults=2)
    store.save("+905551112233", state)
    assert store.get("+905551112233").tour == "Ephesus"
    assert store.get("+905551112233").adults == 2


def test_save_copies_input(store) -> None:
    state = ConversationState(tour="Ephesus")
    store.save("+905551112233", state)
    state.tour = "Mutated"
    assert store.get("+905551112233").tour == "Ephesus"


def test_get_returns_copy_mutations_do_not_affect_store(store) -> None:
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    fetched = store.get("+905551112233")
    fetched.tour = "Mutated"
    fetched.adults = 99
    assert store.get("+905551112233").tour == "Ephesus"
    assert store.get("+905551112233").adults is None


def test_customer_isolation(store) -> None:
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    store.save("+905559999999", ConversationState(tour="Pamukkale"))

    first = store.get("+905551112233")
    second = store.get("+905559999999")

    assert first.tour == "Ephesus"
    assert second.tour == "Pamukkale"

    # Mutating one customer's state never affects the other.
    second.tour = "Overwritten"
    assert store.get("+905551112233").tour == "Ephesus"
    assert store.get("+905559999999").tour == "Pamukkale"


def test_clear_removes_all_states(store) -> None:
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    store.save("+905559999999", ConversationState(tour="Pamukkale"))
    store.clear()
    assert store.get("+905551112233") == ConversationState()
    assert store.get("+905559999999") == ConversationState()


def test_saved_entities_persist(store) -> None:
    from datetime import date

    state = ConversationState(
        tour="Ephesus",
        travel_date=date(2026, 9, 10),
        adults=2,
        children=1,
        cruise_ship="Equinox",
        hotel="Korumar",
        pickup_location="Port",
        preferred_language="English",
    )
    store.save("+905551112233", state)
    loaded = store.get("+905551112233")
    for field in (
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    ):
        assert getattr(loaded, field) == getattr(state, field)


def test_saved_booking_stage_and_flags_persist(store) -> None:
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        booking_stage=BookingStage.READY_FOR_REVIEW,
        needs_human=True,
    )
    store.save("+905551112233", state)
    loaded = store.get("+905551112233")
    assert loaded.booking_stage is BookingStage.READY_FOR_REVIEW
    assert loaded.intent is ConversationIntent.BOOKING_REQUEST
    assert loaded.needs_human is True


def test_phone_keys_distinct_by_exact_normalized_input(store) -> None:
    store.save("+905551112233", ConversationState(tour="A"))
    store.save("+905559999999", ConversationState(tour="B"))
    assert store.get("+905551112233").tour == "A"
    assert store.get("+905559999999").tour == "B"


def test_whitespace_around_phone_is_normalized(store) -> None:
    store.save("  +905551112233  ", ConversationState(tour="Ephesus"))
    assert store.get("+905551112233").tour == "Ephesus"
    assert store.get("  +905551112233  ").tour == "Ephesus"


def test_global_accessor_returns_same_instance() -> None:
    assert get_conversation_store() is get_conversation_store()
    assert isinstance(get_conversation_store(), InMemoryConversationStore)


def test_no_network_or_environment_dependency(store) -> None:
    import os

    snapshot = dict(os.environ)
    store.save("+905551112233", ConversationState())
    store.get("+905551112233")
    assert dict(os.environ) == snapshot
