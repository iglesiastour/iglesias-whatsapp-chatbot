-- 0003: handoff_audit_events
-- Append-only audit trail for operator lifecycle status changes.
-- Idempotent and non-destructive: safe to run multiple times.

CREATE TABLE IF NOT EXISTS handoff_audit_events (
    id UUID PRIMARY KEY,

    handoff_id UUID NOT NULL REFERENCES handoff_requests(id),

    action TEXT NOT NULL,
    previous_status TEXT NOT NULL,
    new_status TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT handoff_audit_events_action_allowed CHECK (
        action IN ('status_changed')
    ),
    CONSTRAINT handoff_audit_events_previous_status_allowed CHECK (
        previous_status IN (
            'pending',
            'in_review',
            'resolved',
            'cancelled'
        )
    ),
    CONSTRAINT handoff_audit_events_new_status_allowed CHECK (
        new_status IN (
            'pending',
            'in_review',
            'resolved',
            'cancelled'
        )
    )
);
