"""Append-only audit repository contract for lifecycle status changes.

Audit records may only be created and listed — never updated, deleted, or
cleared through the production interface.
"""

from abc import ABC, abstractmethod
from uuid import UUID

from app.models.handoff import HandoffStatus
from app.models.handoff_audit import HandoffAuditEvent


class HandoffAuditRepository(ABC):
    @abstractmethod
    def create_status_change(
        self,
        *,
        handoff_id: UUID,
        previous_status: HandoffStatus,
        new_status: HandoffStatus,
    ) -> HandoffAuditEvent:
        """Append one status-change audit event."""
        raise NotImplementedError

    @abstractmethod
    def list_for_handoff(
        self,
        handoff_id: UUID,
    ) -> list[HandoffAuditEvent]:
        """List audit events for a handoff in deterministic order."""
        raise NotImplementedError


class HandoffAuditError(Exception):
    """Raised when audit persistence fails (status may already be updated)."""
