"""Tests for the handoff audit domain model (Phase 7 Step 4)."""

from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.models.handoff import HandoffStatus
from app.models.handoff_audit import (
    HandoffAuditAction,
    HandoffAuditEvent,
)

HANDOFF_ID = UUID("12345678123456781234567812345678")


def _event() -> HandoffAuditEvent:
    return HandoffAuditEvent(
        id=UUID("00000000000000000000000000000001"),
        handoff_id=HANDOFF_ID,
        action=HandoffAuditAction.STATUS_CHANGED,
        previous_status=HandoffStatus.PENDING,
        new_status=HandoffStatus.IN_REVIEW,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_model_is_frozen():
    event = _event()
    assert event.model_config.get("frozen") is True
    with pytest.raises(ValidationError):
        event.new_status = HandoffStatus.RESOLVED  # type: ignore[misc]


def test_action_enum_exact():
    assert HandoffAuditAction.STATUS_CHANGED.value == "status_changed"
    assert {a.value for a in HandoffAuditAction} == {"status_changed"}


def test_created_at_timezone_aware():
    event = _event()
    assert event.created_at.tzinfo is not None


def test_no_customer_pii_fields():
    fields = set(HandoffAuditEvent.model_fields)
    for forbidden in (
        "customer_phone",
        "customer_name",
        "tour",
        "travel_date",
        "adults",
        "booking_stage",
        "reason",
    ):
        assert forbidden not in fields


def test_no_ai_auth_or_internal_fields():
    fields = set(HandoffAuditEvent.model_fields)
    for forbidden in (
        "token",
        "api_key",
        "message",
        "transcript",
        "prompt",
        "reasoning",
        "idempotency_key",
        "ip_address",
        "operator",
        "notes",
    ):
        assert forbidden not in fields


def test_valid_full_event():
    event = _event()
    assert event.handoff_id == HANDOFF_ID
    assert event.previous_status is HandoffStatus.PENDING
    assert event.new_status is HandoffStatus.IN_REVIEW
