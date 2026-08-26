"""Read-only operator-facing handoff review routes.

Protected by the operator auth boundary. Strictly read-only: viewing a
handoff never mutates its status (viewing is not reviewing).
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from pydantic import BaseModel, ConfigDict

from app.db.connection import DatabaseNotConfiguredError
from app.models.handoff import HandoffReason, HandoffStatus
from app.models.handoff_audit import HandoffAuditEvent
from app.models.handoff_review import (
    HandoffReview,
    HandoffReviewListResponse,
    build_handoff_review,
)
from app.repositories.handoff_audit_repository import HandoffAuditError
from app.repositories.handoff_repository import HandoffNotFoundError
from app.repositories.provider import (
    RepositoryConfigurationError,
    get_handoff_audit_repository,
    get_handoff_repository,
)
from app.security.operator_auth import require_operator_credentials
from app.services.handoff_lifecycle_service import (
    HandoffLifecycleService,
    InvalidHandoffTransitionError,
)
from app.services.handoff_review_service import HandoffReviewService


router = APIRouter(
    prefix="/operator/handoffs",
    tags=["operator"],
    dependencies=[Depends(require_operator_credentials)],
)
logger = logging.getLogger(__name__)


class UpdateHandoffStatusRequest(BaseModel):
    """Operator status-transition command. Status only — nothing else."""

    model_config = ConfigDict(extra="forbid")

    status: HandoffStatus


class HandoffAuditListResponse(BaseModel):
    """Paginated-safe collection of audit events (no PII/internal fields)."""

    items: list[HandoffAuditEvent]


@router.get("/{handoff_id}", response_model=HandoffReview)
async def get_handoff_review(handoff_id: UUID) -> HandoffReview:
    """Return the safe human review view for one persisted handoff."""
    try:
        repository = get_handoff_repository()
    except RepositoryConfigurationError:
        # Do not echo invalid backend/config values.
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review service is not configured correctly.",
        ) from None

    service = HandoffReviewService(repository)

    try:
        return service.get_review(handoff_id)
    except HandoffNotFoundError:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Handoff not found.",
        ) from None
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Handoff review lookup failed.")
        # Safe detail only: no SQL/UUID internals/backend leakage.
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None


@router.patch("/{handoff_id}/status", response_model=HandoffReview)
async def update_handoff_status(
    handoff_id: UUID,
    request: UpdateHandoffStatusRequest,
) -> HandoffReview:
    """Apply a human-owned lifecycle status transition (status only)."""
    try:
        repository = get_handoff_repository()
        audit_repository = get_handoff_audit_repository()
    except RepositoryConfigurationError:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review service is not configured correctly.",
        ) from None

    lifecycle = HandoffLifecycleService(repository, audit_repository=audit_repository)

    try:
        updated = lifecycle.transition(handoff_id, request.status)
    except HandoffNotFoundError:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Handoff not found.",
        ) from None
    except InvalidHandoffTransitionError:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail="Invalid handoff status transition.",
        ) from None
    except HandoffAuditError:
        # Audit persistence failed after a successful status update. The status
        # may already be updated; report that the audit service is unavailable
        # rather than implying nothing changed.
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review audit service is unavailable.",
        ) from None
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Handoff status transition failed.")
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None

    return build_handoff_review(updated)


@router.get("", response_model=HandoffReviewListResponse)
async def list_handoffs(
    status: HandoffStatus | None = Query(default=None),
    reason: HandoffReason | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> HandoffReviewListResponse:
    """List safe handoff reviews with optional filters and pagination."""
    try:
        repository = get_handoff_repository()
    except RepositoryConfigurationError:
        # Do not echo invalid backend/config values.
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review service is not configured correctly.",
        ) from None

    service = HandoffReviewService(repository)

    try:
        items = service.list_reviews(
            status=status,
            reason=reason,
            limit=limit,
            offset=offset,
        )
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Handoff listing failed.")
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None

    return HandoffReviewListResponse(
        items=items,
        limit=limit,
        offset=offset,
        count=len(items),
    )


@router.get("/{handoff_id}/audit", response_model=HandoffAuditListResponse)
async def list_handoff_audit(handoff_id: UUID) -> HandoffAuditListResponse:
    """List audit events for a handoff (read-only; 404 if handoff unknown)."""
    try:
        repository = get_handoff_repository()
        audit_repository = get_handoff_audit_repository()
    except RepositoryConfigurationError:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Human review service is not configured correctly.",
        ) from None

    # A handoff must exist; an empty audit list for a nonexistent handoff must
    # still be a 404, not an empty success.
    try:
        exists = repository.get(handoff_id) is not None
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    if not exists:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Handoff not found.",
        ) from None

    try:
        events = audit_repository.list_for_handoff(handoff_id)
    except DatabaseNotConfiguredError:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None
    except HTTPException:
        raise
    except Exception:
        logger.exception("Handoff audit listing failed.")
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Human review service is unavailable.",
        ) from None

    return HandoffAuditListResponse(items=events)
