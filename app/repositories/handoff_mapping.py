"""Pure mapping helpers between HandoffRequest/PersistedHandoff and DB rows.

No database calls in this module — only deterministic value conversion.
Corrupt DB values are rejected (never silently repaired).
"""

from collections.abc import Mapping
from datetime import date
from typing import Any
from uuid import UUID

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.handoff import HandoffRequest, HandoffReason, HandoffStatus, PersistedHandoff


def handoff_request_to_db_values(
    handoff_id: UUID,
    idempotency_key: str,
    request: HandoffRequest,
) -> dict[str, Any]:
    """Flatten a HandoffRequest (with assigned UUID + key) to DB column values.

    Enums are serialized via .value. Timestamps are owned by the DB and are
    intentionally omitted.
    """
    state = request.conversation_state
    return {
        "id": handoff_id,
        "idempotency_key": idempotency_key,
        "customer_phone": request.customer_phone,
        "customer_name": request.customer_name,
        "reason": request.reason.value,
        "status": request.status.value,
        "intent": state.intent.value,
        "tour": state.tour,
        "travel_date": state.travel_date,
        "adults": state.adults,
        "children": state.children,
        "cruise_ship": state.cruise_ship,
        "hotel": state.hotel,
        "pickup_location": state.pickup_location,
        "preferred_language": state.preferred_language,
        "booking_stage": state.booking_stage.value,
        "needs_human": state.needs_human,
    }


def _as_date(value: Any) -> date | None:
    if value is None:
        return None
    assert isinstance(value, date)
    return value


def db_row_to_persisted_handoff(row: Mapping[str, Any]) -> PersistedHandoff:
    """Rebuild a PersistedHandoff from a DB row.

    Created_at/updated_at keys (if present) are ignored. Invalid enum values
    surface as normal enum/Pydantic errors rather than silent repair.
    """
    state = ConversationState(
        intent=ConversationIntent(row["intent"]),
        tour=row["tour"],  # type: ignore[arg-type]
        travel_date=_as_date(row["travel_date"]),
        adults=row["adults"],  # type: ignore[arg-type]
        children=row["children"],  # type: ignore[arg-type]
        cruise_ship=row["cruise_ship"],  # type: ignore[arg-type]
        hotel=row["hotel"],  # type: ignore[arg-type]
        pickup_location=row["pickup_location"],  # type: ignore[arg-type]
        preferred_language=row["preferred_language"],  # type: ignore[arg-type]
        booking_stage=BookingStage(row["booking_stage"]),
        needs_human=bool(row["needs_human"]),
    )

    return PersistedHandoff(
        id=UUID(str(row["id"])),
        idempotency_key=str(row["idempotency_key"]),
        customer_phone=row["customer_phone"],  # type: ignore[arg-type]
        customer_name=row["customer_name"],  # type: ignore[arg-type]
        reason=HandoffReason(row["reason"]),
        status=HandoffStatus(row["status"]),
        conversation_state=state,
    )
