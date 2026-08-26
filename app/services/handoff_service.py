"""Idempotent human-handoff creation service.

Deterministic application service: decides whether a handoff is needed from
verified conversation state, derives an idempotency key, and creates the
handoff exactly once per logical review state. No AI, no notifications,
no route integration in this step.
"""

from app.models.conversation import ConversationState
from app.models.handoff import (
    PersistedHandoff,
    build_handoff_idempotency_key,
    create_handoff_request,
    determine_handoff_reason,
)
from app.repositories.handoff_repository import HandoffRepository


class HandoffService:
    def __init__(self, repository: HandoffRepository):
        self._repository = repository

    def ensure_handoff(
        self,
        customer_phone: str,
        state: ConversationState,
        customer_name: str | None = None,
    ) -> PersistedHandoff | None:
        """Ensure exactly one handoff exists for this logical review state.

        Returns None when no handoff is warranted. Existing handoffs (any
        status) are returned unchanged; status transitions are owned by
        humans/backend and are never reset here.
        """
        reason = determine_handoff_reason(state)
        if reason is None:
            return None

        idempotency_key = build_handoff_idempotency_key(
            customer_phone,
            state,
            reason,
        )

        existing = self._repository.get_by_idempotency_key(idempotency_key)
        if existing is not None:
            return existing

        request = create_handoff_request(
            customer_phone=customer_phone,
            state=state,
            reason=reason,
            customer_name=customer_name,
        )

        return self._repository.create(request, idempotency_key)
