-- 0002: handoff_requests
-- Persistent human-review handoff records derived from ConversationState.
-- Idempotent and non-destructive: safe to run multiple times.

CREATE TABLE IF NOT EXISTS handoff_requests (
    id UUID PRIMARY KEY,

    idempotency_key TEXT NOT NULL UNIQUE,

    customer_phone TEXT NOT NULL,
    customer_name TEXT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,

    intent TEXT NOT NULL,
    tour TEXT NULL,
    travel_date DATE NULL,

    adults INTEGER NULL,
    children INTEGER NULL,

    cruise_ship TEXT NULL,
    hotel TEXT NULL,
    pickup_location TEXT NULL,
    preferred_language TEXT NULL,

    booking_stage TEXT NOT NULL,
    needs_human BOOLEAN NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT handoff_requests_reason_allowed CHECK (
        reason IN (
            'booking_review',
            'human_request',
            'complaint',
            'cancellation_request',
            'existing_booking',
            'safety_escalation'
        )
    ),
    CONSTRAINT handoff_requests_status_allowed CHECK (
        status IN (
            'pending',
            'in_review',
            'resolved',
            'cancelled'
        )
    ),
    CONSTRAINT handoff_requests_intent_allowed CHECK (
        intent IN (
            'greeting',
            'general_question',
            'tour_information',
            'price_request',
            'availability_request',
            'booking_request',
            'existing_booking',
            'cancellation_request',
            'complaint',
            'human_request',
            'unsupported'
        )
    ),
    CONSTRAINT handoff_requests_booking_stage_allowed CHECK (
        booking_stage IN (
            'none',
            'collecting_details',
            'ready_for_review',
            'human_review',
            'confirmed',
            'cancelled'
        )
    ),
    CONSTRAINT handoff_requests_adults_range CHECK (
        adults IS NULL OR (adults >= 1 AND adults <= 100)
    ),
    CONSTRAINT handoff_requests_children_range CHECK (
        children IS NULL OR (children >= 0 AND children <= 100)
    )
);
