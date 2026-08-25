"""Conversation repository contract.

Stable application-facing storage abstraction: business logic depends on
this interface, not on a concrete implementation (in-memory today,
PostgreSQL/Neon later).
"""

from abc import ABC, abstractmethod

from app.models.conversation import ConversationState


class ConversationRepository(ABC):

    @abstractmethod
    def get(self, customer_phone: str) -> ConversationState:
        """Return stored conversation state or a new default state."""
        raise NotImplementedError

    @abstractmethod
    def save(
        self,
        customer_phone: str,
        state: ConversationState,
    ) -> None:
        """Persist the conversation state for a customer."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Clear repository state where supported."""
        raise NotImplementedError
