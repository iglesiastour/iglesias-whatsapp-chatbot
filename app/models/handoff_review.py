"""Human review read model for persisted handoffs.

Represents exactly what a trusted human operator is allowed to see: structured
booking/conversation snapshot data only. Internal AI/security/storage details
(idempotency keys, prompts, safety pattern matches, provider responses, raw
messages, DB configuration) are deliberately excluded.
"""

from datetime import date
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.conversation import BookingStage, ConversationIntent
from app.models.handoff import HandoffReason, HandoffStatus, PersistedHandoff


class HandoffReview(BaseModel):
    """Immutable human-facing view of a persisted handoff."""

    model_config = ConfigDict(frozen=True)

    handoff_id: UUID
    customer_phone: str
    customer_name: str | None = None
    reason: HandoffReason
    status: HandoffStatus

    intent: ConversationIntent
    booking_stage: BookingStage
    needs_human: bool = False

    tour: str | None = None
    travel_date: date | None = None
    adults: int | None = None
    children: int | None = None
    cruise_ship: str | None = None
    hotel: str | None = None
    pickup_location: str | None = None
    preferred_language: str | None = None


def build_handoff_review(handoff: PersistedHandoff) -> HandoffReview:
    """Map a persisted handoff (+ its snapshotted state) to a review view.

    Pure function: no repository calls, no input mutation. The review always
    reflects the handoff's own conversation snapshot, never live state.
    """
    state = handoff.conversation_state
    return HandoffReview(
        handoff_id=handoff.id,
        customer_phone=handoff.customer_phone,
        customer_name=handoff.customer_name,
        reason=handoff.reason,
        status=handoff.status,
        intent=state.intent,
        booking_stage=state.booking_stage,
        needs_human=state.needs_human,
        tour=state.tour,
        travel_date=state.travel_date,
        adults=state.adults,
        children=state.children,
        cruise_ship=state.cruise_ship,
        hotel=state.hotel,
        pickup_location=state.pickup_location,
        preferred_language=state.preferred_language,
    )


class HandoffReviewListResponse(BaseModel):
    """Paginated collection of safe handoff reviews."""

    items: list[HandoffReview]
    limit: int
    offset: int
    count: int


def build_handoff_review_summary(review: HandoffReview) -> str:
    """Deterministic human-friendly text summary; unknown fields omitted."""
    lines = [
        "Human review",
        f"- Reason: {review.reason.value}",
        f"- Status: {review.status.value}",
    ]
    if review.customer_name:
        lines.append(f"- Customer: {review.customer_name}")
    lines.append(f"- Phone: {review.customer_phone}")
    if review.tour:
        lines.append(f"- Tour: {review.tour}")
    if review.travel_date:
        lines.append(f"- Travel date: {review.travel_date.isoformat()}")
    if review.adults is not None:
        lines.append(f"- Adults: {review.adults}")
    if review.children is not None:
        lines.append(f"- Children: {review.children}")
    if review.cruise_ship:
        lines.append(f"- Cruise ship: {review.cruise_ship}")
    if review.hotel:
        lines.append(f"- Hotel: {review.hotel}")
    if review.pickup_location:
        lines.append(f"- Pickup location: {review.pickup_location}")
    if review.preferred_language:
        lines.append(f"- Preferred language: {review.preferred_language}")
    lines.append(f"- Intent: {review.intent.value}")
    lines.append(f"- Booking stage: {review.booking_stage.value}")
    lines.append(f"- Needs human: {'yes' if review.needs_human else 'no'}")
    return "\n".join(lines)
