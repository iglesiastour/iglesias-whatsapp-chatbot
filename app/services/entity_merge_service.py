"""Deterministic merge of structured extractions into conversation state.

Merge semantics: a non-None extracted value overwrites the state value;
None (or blank-after-trim strings) means "no update" — never a deletion.
"""

from app.models.conversation import BookingStage, ConversationIntent, ConversationState
from app.models.extraction import ExtractedEntities, StructuredExtraction

# Editable string entity fields that are trimmed before storage.
_TRIMMED_FIELDS: tuple[str, ...] = (
    "tour",
    "cruise_ship",
    "hotel",
    "pickup_location",
    "preferred_language",
)


def _effective_updates(
    entities: ExtractedEntities,
) -> dict[str, object]:
    """Compute non-None entity updates with string trimming applied."""
    updates: dict[str, object] = {}
    for field in type(entities).model_fields:
        value = getattr(entities, field)
        if value is None:
            continue
        if field in _TRIMMED_FIELDS:
            assert isinstance(value, str)
            trimmed = value.strip()
            if not trimmed:
                continue  # blank after trim == absent; preserve existing value
            updates[field] = trimmed
        else:
            updates[field] = value
    return updates


def merge_extraction_into_state(
    state: ConversationState,
    extraction: StructuredExtraction,
) -> ConversationState:
    """Return a NEW ConversationState with extracted entities merged in.

    The incoming state is never mutated. Intent, needs_human and booking
    stage are preserved except for the deterministic stage-recalculation
    rules below.
    """
    merged = state.model_copy(update=_effective_updates(extraction.entities))

    # Authoritative business stages are never recalculated from data.
    if state.booking_stage in (BookingStage.CONFIRMED, BookingStage.CANCELLED):
        return merged

    # HUMAN_REVIEW is sticky in this phase.
    if state.booking_stage is BookingStage.HUMAN_REVIEW:
        return merged

    # Stage recalculation from completeness: an active BOOKING_REQUEST, or a
    # conversation already mid-collection (COLLECTING_DETAILS) whose required
    # fields just became complete (multi-turn extraction), may promote to
    # READY_FOR_REVIEW. NONE stays NONE for non-booking intents.
    if (
        merged.intent is ConversationIntent.BOOKING_REQUEST
        or state.booking_stage is BookingStage.COLLECTING_DETAILS
    ):
        if merged.missing_booking_fields():
            merged = merged.model_copy(
                update={"booking_stage": BookingStage.COLLECTING_DETAILS}
            )
        else:
            merged = merged.model_copy(
                update={"booking_stage": BookingStage.READY_FOR_REVIEW}
            )

    return merged
