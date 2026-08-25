"""Temporary in-memory conversation state store.

DEVELOPMENT/TRANSITION ONLY: this store exists so conversation memory works
end-to-end before a real database (e.g., PostgreSQL) replaces it. Its small
interface (get/save/clear) is intentionally easy to swap later.
"""

from app.models.conversation import ConversationState


def _normalize_phone(customer_phone: str) -> str:
    """Normalize the customer phone used as the store key."""
    return " ".join(customer_phone.split())


class InMemoryConversationStore:
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


def get_conversation_store() -> InMemoryConversationStore:
    """Return the process-wide conversation store instance."""
    return _store
