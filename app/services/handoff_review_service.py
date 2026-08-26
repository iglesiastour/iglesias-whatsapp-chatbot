"""Read-only human review service for persisted handoffs.

Viewing a handoff review is a read operation: it never claims, starts, or
mutates the review. Status transitions remain human/backend-owned via the
lifecycle service (Step 5).
"""

from uuid import UUID

from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff
from app.models.handoff_review import HandoffReview, build_handoff_review
from app.repositories.handoff_repository import (
    HandoffNotFoundError,
    HandoffRepository,
)


class HandoffReviewService:
    def __init__(self, repository: HandoffRepository):
        self._repository = repository

    def get_review(self, handoff_id: UUID) -> HandoffReview:
        """Return the immutable review view for a persisted handoff."""
        persisted = self._repository.get(handoff_id)
        if persisted is None:
            raise HandoffNotFoundError()
        return build_handoff_review(persisted)

    def list_reviews(
        self,
        *,
        status: HandoffStatus | None = None,
        reason: HandoffReason | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[HandoffReview]:
        """List safe review views (read-only; never mutates handoffs)."""
        persisted_items = self._repository.list_handoffs(
            status=status,
            reason=reason,
            limit=limit,
            offset=offset,
        )
        return [build_handoff_review(item) for item in persisted_items]
