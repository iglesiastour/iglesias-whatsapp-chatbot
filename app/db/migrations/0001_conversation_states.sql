-- 0001: conversation_states
-- Persistent conversation state for the WhatsApp chatbot.
-- Idempotent and non-destructive: safe to run multiple times.

CREATE TABLE IF NOT EXISTS conversation_states (
    customer_phone TEXT PRIMARY KEY,

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
    needs_human BOOLEAN NOT NULL DEFAULT FALSE,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT conversation_states_adults_range CHECK (
        adults IS NULL OR (adults >= 1 AND adults <= 100)
    ),
    CONSTRAINT conversation_states_children_range CHECK (
        children IS NULL OR (children >= 0 AND children <= 100)
    ),
    CONSTRAINT conversation_states_intent_allowed CHECK (
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
    CONSTRAINT conversation_states_booking_stage_allowed CHECK (
        booking_stage IN (
            'none',
            'collecting_details',
            'ready_for_review',
            'human_review',
            'confirmed',
            'cancelled'
        )
    )
);
