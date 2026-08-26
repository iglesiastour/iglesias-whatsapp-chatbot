"""In-memory append-only handoff audit repository (test/dev parity)."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.models.handoff import HandoffStatus
from app.models.handoff_audit import HandoffAuditAction, HandoffAuditEvent
from app.repositories.handoff_audit_repository import HandoffAuditRepository


class InMemoryHandoffAuditRepository(HandoffAuditRepository):
    def __init__(self) -> None:
        self._events: list[HandoffAuditEvent] = []

    def create_status_change(
        self,
        *,
        handoff_id: UUID,
        previous_status: HandoffStatus,
        new_status: HandoffStatus,
    ) -> HandoffAuditEvent:
        event = HandoffAuditEvent(
            id=uuid4(),
            handoff_id=handoff_id,
            action=HandoffAuditAction.STATUS_CHANGED,
            previous_status=previous_status,
            new_status=new_status,
            created_at=datetime.now(timezone.utc),
        )
        self._events.append(event.model_copy(deep=True))
        return event.model_copy(deep=True)

    def list_for_handoff(
        self,
        handoff_id: UUID,
    ) -> list[HandoffAuditEvent]:
        matches = [e for e in self._events if e.handoff_id == handoff_id]
        return [e.model_copy(deep=True) for e in matches]

    def reset(self) -> None:
        """Test-only reset; not part of the production interface."""
        self._events.clear()
