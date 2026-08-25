"""Pure mapping helpers between ConversationState and DB row values.

No database calls in this module — only deterministic value conversion.
"""

from collections.abc import Mapping
from datetime import date

from app.models.conversation import BookingStage, ConversationIntent, ConversationState


def state_to_db_values(
    customer_phone: str,
    state: ConversationState,
) -> dict[str, object]:
    """Map a ConversationState to DB-facing column values (no timestamps)."""
    return {
        "customer_phone": customer_phone,
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


def db_row_to_state(row: Mapping[str, object]) -> ConversationState:
    """Construct a ConversationState from a mapping-like DB row.

    Column order is irrelevant; created_at/updated_at keys are ignored.
    Invalid intent/stage values surface as normal enum/Pydantic errors.
    """
    return ConversationState(
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


def _as_date(value: object) -> date | None:
    if value is None:
        return None
    assert isinstance(value, date)
    return value
