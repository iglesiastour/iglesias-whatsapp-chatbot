"""Temporary in-memory conversation state repository.

DEVELOPMENT/TRANSITION ONLY: this implementation exists so conversation
memory works end-to-end before a real database (e.g., PostgreSQL) replaces
it via the ConversationRepository interface.
"""

from app.models.conversation import ConversationState
from app.repositories.conversation_repository import ConversationRepository
from app.services.phone_normalizer import normalize_customer_phone


def _normalize_phone(customer_phone: str) -> str:
    """Normalize the customer phone used as the storage key."""
    return normalize_customer_phone(customer_phone)


class InMemoryConversationStore(ConversationRepository):
    """In-memory ConversationState store keyed by normalized customer phone."""

    def __init__(self) -> None:
        self._states: dict[str, ConversationState] = {}

    def get(self, customer_phone: str) -> ConversationState:
        """Return a COPY of the stored state, or a NEW default state."""
        key = _normalize_phone(customer_phone)
        stored = self._states.get(key)
        if stored is None:
            return ConversationState()
        return stored.model_copy()

    def save(self, customer_phone: str, state: ConversationState) -> None:
        """Store a COPY of the given state under the normalized phone key."""
        key = _normalize_phone(customer_phone)
        self._states[key] = state.model_copy()

    def clear(self) -> None:
        """Remove all stored states (test/ops helper)."""
        self._states.clear()


_store = InMemoryConversationStore()


def get_conversation_store() -> ConversationRepository:
    """Return the process-wide conversation store instance."""
    return _store