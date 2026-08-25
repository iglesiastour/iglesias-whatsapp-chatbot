"""Repository selection factory.

Selects the active ConversationRepository implementation from configuration.
Default is the safe in-memory backend; postgres must be explicitly chosen.
"""

from app.config import settings
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.postgres_conversation_repository import (
    PostgresConversationRepository,
)
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


_memory_instance = InMemoryConversationStore()


def _memory_singleton() -> InMemoryConversationStore:
    return _memory_instance
