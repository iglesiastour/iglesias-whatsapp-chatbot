"""Human handoff / review domain models.

Application-owned contract only: no notifications, WhatsApp/Kommo/n8n
integration, or AI involvement in persistence/decisions.
"""

from datetime import date
from enum import StrEnum
from hashlib import sha256
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.services.phone_normalizer import normalize_customer_phone


class HandoffReason(StrEnum):
    BOOKING_REVIEW = "booking_review"
    HUMAN_REQUEST = "human_request"
    COMPLAINT = "complaint"
    CANCELLATION_REQUEST = "cancellation_request"
    EXISTING_BOOKING = "existing_booking"
    SAFETY_ESCALATION = "safety_escalation"


class HandoffStatus(StrEnum):
    # PENDING means the application created a review item; it does NOT mean a
    # human has seen it. IN_REVIEW/RESOLVED/CANCELLED are future backend/human-
    # owned transitions; the AI must never set them.
    PENDING = "pending"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"
    CANCELLED = "cancelled"


def _clean_optional_text(value: str | None) -> str | None:
    """Strip + collapse whitespace, preserve capitalization; blank -> None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return " ".join(stripped.split())


def determine_handoff_reason(state: ConversationState) -> HandoffReason | None:
    """Deterministically derive a handoff reason from conversation state.

    Priority order (first match wins):
      1. cancellation intent
      2. complaint intent
      3. human-request intent
      4. existing booking that requires human
      5. booking ready for review
      6. generic human flag
    Ordinary collecting-details bookings and tour-information questions never
    create a handoff. Authoritative CONFIRMED/CANCELLED stages are not treated
    as new requests by themselves.
    """
    if state.intent is ConversationIntent.CANCELLATION_REQUEST:
        return HandoffReason.CANCELLATION_REQUEST
    if state.intent is ConversationIntent.COMPLAINT:
        return HandoffReason.COMPLAINT
    if state.intent is ConversationIntent.HUMAN_REQUEST:
        return HandoffReason.HUMAN_REQUEST
    if (
        state.intent is ConversationIntent.EXISTING_BOOKING
        and state.requires_human
    ):
        return HandoffReason.EXISTING_BOOKING
    if state.booking_stage is BookingStage.READY_FOR_REVIEW:
        return HandoffReason.BOOKING_REVIEW
    if state.requires_human:
        return HandoffReason.SAFETY_ESCALATION
    return None


class HandoffRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    customer_phone: str
    customer_name: str | None = None
    reason: HandoffReason
    status: HandoffStatus = HandoffStatus.PENDING
    conversation_state: ConversationState

    @field_validator("customer_phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        normalized = normalize_customer_phone(value)
        if not normalized:
            raise ValueError("customer_phone must not be blank.")
        return normalized

    @field_validator("customer_name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)


class PersistedHandoff(BaseModel):
    """A handoff that has been assigned a persistent identity.

    Distinct from HandoffRequest (the creation command/snapshot): this is the
    form returned by the repository after persistence and carries the
    assigned UUID. It intentionally omits timestamps so it stays free of
    DB-owned bookkeeping columns.
    """

    model_config = ConfigDict(frozen=True)

    id: UUID
    idempotency_key: str
    customer_phone: str
    customer_name: str | None = None
    reason: HandoffReason
    status: HandoffStatus = HandoffStatus.PENDING
    conversation_state: ConversationState

    @field_validator("customer_phone")
    @classmethod
    def _validate_phone(cls, value: str) -> str:
        normalized = normalize_customer_phone(value)
        if not normalized:
            raise ValueError("customer_phone must not be blank.")
        return normalized

    @field_validator("customer_name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str | None:
        return _clean_optional_text(value)


def create_handoff_request(
    customer_phone: str,
    state: ConversationState,
    reason: HandoffReason,
    customer_name: str | None = None,
) -> HandoffRequest:
    """Create a handoff request with an independent deep snapshot of state."""
    return HandoffRequest(
        customer_phone=normalize_customer_phone(customer_phone),
        customer_name=_clean_optional_text(customer_name),
        reason=reason,
        status=HandoffStatus.PENDING,
        conversation_state=state.model_copy(deep=True),
    )


def build_handoff_idempotency_key(
    customer_phone: str,
    state: ConversationState,
    reason: HandoffReason,
) -> str:
    """Deterministic SHA-256 idempotency key for a logical handoff review.

    Identity is derived only from core booking identity fields. Descriptive /
    optional fields (customer_name, children, hotel, cruise_ship, pickup
    location, preferred_language) and AI output are intentionally excluded so
    that cosmetic changes do not fork duplicate review tasks.

    The phone is normalized with the shared helper before hashing, so the
    raw phone never appears in the key.
    """
    phone = normalize_customer_phone(customer_phone)
    parts = [
        phone,
        reason.value,
        state.booking_stage.value,
        state.tour or "",
        state.travel_date.isoformat() if state.travel_date else "",
        str(state.adults) if state.adults is not None else "",
    ]
    payload = "|".join(parts)
    return sha256(payload.encode("utf-8")).hexdigest()

