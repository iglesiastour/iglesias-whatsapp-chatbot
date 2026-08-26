"""Repository selection factory.

Selects the active ConversationRepository implementation from configuration.
Default is the safe in-memory backend; postgres must be explicitly chosen.
"""

from app.config import settings
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.handoff_repository import HandoffRepository
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.repositories.postgres_conversation_repository import (
    PostgresConversationRepository,
)
from app.repositories.postgres_handoff_repository import PostgresHandoffRepository
from app.services.conversation_store import InMemoryConversationStore


class RepositoryConfigurationError(Exception):
    """Raised when repository backend configuration is invalid."""


def get_conversation_repository() -> ConversationRepository:
    """Return the configured conversation repository implementation."""
    backend = settings.conversation_repository_backend.strip().lower()

    if backend == "memory":
        return _memory_singleton()

    if backend == "postgres":
        return PostgresConversationRepository()

    raise RepositoryConfigurationError(
        "Unsupported conversation repository backend."
    )


def get_handoff_repository() -> HandoffRepository:
    """Return the configured handoff repository implementation."""
    backend = settings.conversation_repository_backend.strip().lower()

    if backend == "memory":
        return InMemoryHandoffRepository()

    if backend == "postgres":
        return PostgresHandoffRepository()

    raise RepositoryConfigurationError("Unsupported handoff repository backend.")


_memory_instance = InMemoryConversationStore()


def _memory_singleton() -> InMemoryConversationStore:
    return _memory_instance
