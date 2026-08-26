"""Human-owned handoff lifecycle transitions.

Lifecycle semantics (human/backend owned — the AI and customer message route
must never mutate handoff status):

- PENDING:   the application created the review item (no human has seen it)
- IN_REVIEW: a trusted backend/human action marks a human as actively reviewing
- RESOLVED:  trusted backend/human action marks review complete
- CANCELLED: trusted backend/human action closes the handoff without completion

These statuses describe ONLY the review workflow. They MUST NOT be interpreted
as customer booking status: HandoffStatus.RESOLVED does NOT mean booking
confirmed, payment received, availability confirmed, or tour completed. The
handoff lifecycle is fully independent of BookingStage.
"""

from uuid import UUID

from app.models.handoff import HandoffStatus, PersistedHandoff
from app.repositories.handoff_audit_repository import (
    HandoffAuditError,
    HandoffAuditRepository,
)
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
)


class InvalidHandoffTransitionError(Exception):
    """Raised when a handoff status transition is not permitted."""

    def __init__(self) -> None:
        super().__init__("Invalid handoff status transition.")


_ALLOWED_TRANSITIONS: dict[HandoffStatus, set[HandoffStatus]] = {
    HandoffStatus.PENDING: {
        HandoffStatus.IN_REVIEW,
        HandoffStatus.RESOLVED,
        HandoffStatus.CANCELLED,
    },
    HandoffStatus.IN_REVIEW: {
        HandoffStatus.RESOLVED,
        HandoffStatus.CANCELLED,
    },
    # Terminal states: no outgoing transitions.
    HandoffStatus.RESOLVED: set(),
    HandoffStatus.CANCELLED: set(),
}


def validate_handoff_transition(
    current: HandoffStatus,
    target: HandoffStatus,
) -> None:
    """Pure deterministic validator.

    Allowed or same-status transitions return None; anything else raises
    InvalidHandoffTransitionError.
    """
    if current is target:
        return None
    if target in _ALLOWED_TRANSITIONS[current]:
        return None
    raise InvalidHandoffTransitionError()


class HandoffLifecycleService:
    def __init__(
        self,
        repository: HandoffRepository,
        audit_repository: HandoffAuditRepository | None = None,
    ):
        self._repository = repository
        self._audit_repository = audit_repository

    def transition(
        self,
        handoff_id: UUID,
        target_status: HandoffStatus,
    ) -> PersistedHandoff:
        """Apply a human-owned status transition to a persisted handoff.

        Same-status transitions are idempotent and never produce audit events.
        If an audit repository is configured and audit persistence fails AFTER
        the status update, HandoffAuditError is raised — the status may already
        be updated. No compensating rollback is attempted.
        """
        existing = self._repository.get(handoff_id)
        if existing is None:
            raise HandoffNotFoundError()

        validate_handoff_transition(existing.status, target_status)

        if existing.status is target_status:
            return existing

        updated = self._repository.update_status(handoff_id, target_status)

        if self._audit_repository is not None:
            self._audit_repository.create_status_change(
                handoff_id=handoff_id,
                previous_status=existing.status,
                new_status=updated.status,
            )

        return updated
