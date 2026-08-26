"""Minimal append-only audit trail for human/operator lifecycle changes.

Records ONLY trusted structured lifecycle events (status transitions).
Deliberately excludes customer PII, booking details, raw request bodies,
auth tokens, IP addresses, AI output/prompts, idempotency keys, and DB
internals.
"""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.handoff import HandoffStatus


class HandoffAuditAction(StrEnum):
    STATUS_CHANGED = "status_changed"


class HandoffAuditEvent(BaseModel):
    """One immutable lifecycle status-change event."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    handoff_id: UUID
    action: HandoffAuditAction
    previous_status: HandoffStatus
    new_status: HandoffStatus
    created_at: datetime
