"""Repository contract tests using the in-memory implementation."""

import os

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.repositories.conversation_repository import ConversationRepository
from app.services.conversation_store import (
    InMemoryConversationStore,
    get_conversation_store,
)


def test_implementation_is_subclass_of_repository() -> None:
    assert issubclass(InMemoryConversationStore, ConversationRepository)
    assert isinstance(InMemoryConversationStore(), ConversationRepository)


def test_missing_customer_returns_default_state() -> None:
    store = InMemoryConversationStore()
    state = store.get("+905551112233")
    assert isinstance(state, ConversationState)
    assert state == ConversationState()


def test_save_get_roundtrip_preserves_fields() -> None:
    store = InMemoryConversationStore()
    state = ConversationState(
        intent=ConversationIntent.BOOKING_REQUEST,
        tour="Ephesus",
        booking_stage=BookingStage.COLLECTING_DETAILS,
        needs_human=False,
    )
    store.save("+905551112233", state)
    loaded = store.get("+905551112233")
    for field in ("intent", "tour", "booking_stage", "needs_human"):
        assert getattr(loaded, field) == getattr(state, field)


def test_get_returns_copy_not_internal_object() -> None:
    store = InMemoryConversationStore()
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    first = store.get("+905551112233")
    second = store.get("+905551112233")
    assert first is not second

    first.tour = "Mutated"
    assert store.get("+905551112233").tour == "Ephesus"


def test_save_stores_copy_of_input() -> None:
    store = InMemoryConversationStore()
    state = ConversationState(tour="Ephesus")
    store.save("+905551112233", state)

    state.tour = "Mutated"
    assert store.get("+905551112233").tour == "Ephesus"


def test_customer_isolation() -> None:
    store = InMemoryConversationStore()
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    store.save("+905559999999", ConversationState(tour="Pamukkale"))

    assert store.get("+905551112233").tour == "Ephesus"
    assert store.get("+905559999999").tour == "Pamukkale"


def test_clear_removes_all_states() -> None:
    store = InMemoryConversationStore()
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    store.clear()
    assert store.get("+905551112233") == ConversationState()


def test_no_network_or_environment_dependency() -> None:
    snapshot = dict(os.environ)
    store = InMemoryConversationStore()
    store.save("+905551112233", ConversationState(tour="Ephesus"))
    store.get("+905551112233")
    assert dict(os.environ) == snapshot


def test_accessor_returns_repository_contract() -> None:
    repository = get_conversation_store()
    assert isinstance(repository, ConversationRepository)
    assert get_conversation_store() is repository  # stable singleton


def test_module_has_only_storage_concerns() -> None:
    import sys

    module = sys.modules[InMemoryConversationStore.__module__]
    for forbidden in ("OpenRouterProvider", "SafeAIService", "router"):
        assert not hasattr(module, forbidden)


def test_no_openrouter_dependency() -> None:
    import app.services.conversation_store as module

    source_names = dir(module)
    assert "httpx" not in source_names
    assert not any("openrouter" in name.lower() for name in source_names)


def test_no_safe_ai_service_dependency() -> None:
    import app.services.conversation_store as module

    assert not hasattr(module, "SafeAIService")
    assert not hasattr(module, "INPUT_SAFETY_REPLY")


def test_no_route_dependency() -> None:
    import app.services.conversation_store as module

    assert not hasattr(module, "router")
    assert not hasattr(module, "process_message")
