"""Temporary in-memory handoff repository.

DEVELOPMENT/TRANSITION ONLY: provides test/dev parity for the
HandoffRepository contract without a real database. Not wired into
production routing yet.
"""

from uuid import UUID, uuid4

from app.models.handoff import HandoffRequest, HandoffStatus, PersistedHandoff
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
    HandoffRepositoryDuplicateError,
)


class InMemoryHandoffRepository(HandoffRepository):
    """In-memory PersistedHandoff store keyed by id with a secondary
    idempotency_key index."""

    def __init__(self) -> None:
        self._store: dict[UUID, PersistedHandoff] = {}
        self._by_key: dict[str, UUID] = {}

    def create(
        self,
        request: HandoffRequest,
        idempotency_key: str,
    ) -> PersistedHandoff:
        if idempotency_key in self._by_key:
            raise HandoffRepositoryDuplicateError(
                "Handoff already exists for idempotency_key: " + idempotency_key
            )

        handoff_id = uuid4()
        persisted = PersistedHandoff(
            id=handoff_id,
            idempotency_key=idempotency_key,
            customer_phone=request.customer_phone,
            customer_name=request.customer_name,
            reason=request.reason,
            status=request.status,
            conversation_state=request.conversation_state.model_copy(deep=True),
        )
        # Store and return an independent copy so later source mutation can
        # never affect what is returned.
        self._store[handoff_id] = persisted.model_copy(deep=True)
        self._by_key[idempotency_key] = handoff_id
        return self._store[handoff_id].model_copy(deep=True)

    def get(self, handoff_id: UUID) -> PersistedHandoff | None:
        stored = self._store.get(handoff_id)
        if stored is None:
            return None
        return stored.model_copy(deep=True)

    def get_by_idempotency_key(
        self,
        idempotency_key: str,
    ) -> PersistedHandoff | None:
        handoff_id = self._by_key.get(idempotency_key)
        if handoff_id is None:
            return None
        return self.get(handoff_id)

    def list_handoffs(
        self,
        *,
        status: HandoffStatus | None = None,
        reason=None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PersistedHandoff]:
        """Filter by status/reason in stable insertion order, then paginate."""
        from app.models.handoff import HandoffReason

        matches = [
            stored
            for stored in self._store.values()
            if (status is None or stored.status is status)
            and (reason is None or stored.reason is reason)
        ]
        page = matches[offset : offset + limit]
        return [stored.model_copy(deep=True) for stored in page]

    def update_status(
        self,
        handoff_id: UUID,
        status: HandoffStatus,
    ) -> PersistedHandoff:
        stored = self._store.get(handoff_id)
        if stored is None:
            raise HandoffNotFoundError()

        updated = stored.model_copy(deep=True, update={"status": status})
        self._store[handoff_id] = updated
        return updated.model_copy(deep=True)

    def reset(self) -> None:
        """Test-only reset; not part of the production interface."""
        self._store.clear()
        self._by_key.clear()
