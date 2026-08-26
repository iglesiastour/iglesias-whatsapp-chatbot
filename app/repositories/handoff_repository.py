"""Handoff repository contract.

Stable application-facing storage abstraction for human handoff records.
Business logic depends on this interface, not on a concrete implementation.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

from app.models.handoff import HandoffRequest, PersistedHandoff

if TYPE_CHECKING:
    from app.models.handoff import HandoffStatus


class HandoffRepository(ABC):
    @abstractmethod
    def create(
        self,
        request: HandoffRequest,
        idempotency_key: str,
    ) -> PersistedHandoff:
        """Persist a handoff request with the given idempotency key."""
        raise NotImplementedError

    @abstractmethod
    def get(self, handoff_id: UUID) -> PersistedHandoff | None:
        """Return the persisted handoff for an id, or None if not found."""
        raise NotImplementedError

    @abstractmethod
    def get_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> PersistedHandoff | None:
        """Return an existing persisted handoff for a key, or None."""
        raise NotImplementedError

    @abstractmethod
    def update_status(
        self,
        handoff_id: UUID,
        status: "HandoffStatus",
    ) -> PersistedHandoff:
        """Update ONLY the status of a persisted handoff."""
        raise NotImplementedError


class HandoffRepositoryDuplicateError(Exception):
    """Raised when creating a handoff whose idempotency key already exists."""


class HandoffNotFoundError(Exception):
    """Raised when a handoff record does not exist for the given identity."""

    def __init__(self) -> None:
        super().__init__("Handoff not found.")


