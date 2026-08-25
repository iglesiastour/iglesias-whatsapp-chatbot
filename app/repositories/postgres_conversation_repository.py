"""PostgreSQL-backed ConversationRepository implementation.

Uses explicit parameterized SQL through the shared database_connection()
helper. Not yet selected by the application — the in-memory store remains
active until this repository is wired in.
"""

from psycopg.rows import dict_row

from app.db.connection import database_connection
from app.models.conversation import ConversationState
from app.repositories.conversation_mapping import (
    db_row_to_state,
    state_to_db_values,
)
from app.repositories.conversation_repository import ConversationRepository
from app.services.phone_normalizer import normalize_customer_phone

_SELECT_SQL = """
SELECT
    customer_phone,
    intent,
    tour,
    travel_date,
    adults,
    children,
    cruise_ship,
    hotel,
    pickup_location,
    preferred_language,
    booking_stage,
    needs_human
FROM conversation_states
WHERE customer_phone = %s
"""

_UPSERT_SQL = """
INSERT INTO conversation_states (
    customer_phone,
    intent,
    tour,
    travel_date,
    adults,
    children,
    cruise_ship,
    hotel,
    pickup_location,
    preferred_language,
    booking_stage,
    needs_human
)
VALUES (
    %(customer_phone)s,
    %(intent)s,
    %(tour)s,
    %(travel_date)s,
    %(adults)s,
    %(children)s,
    %(cruise_ship)s,
    %(hotel)s,
    %(pickup_location)s,
    %(preferred_language)s,
    %(booking_stage)s,
    %(needs_human)s
)
ON CONFLICT (customer_phone)
DO UPDATE SET
    intent = EXCLUDED.intent,
    tour = EXCLUDED.tour,
    travel_date = EXCLUDED.travel_date,
    adults = EXCLUDED.adults,
    children = EXCLUDED.children,
    cruise_ship = EXCLUDED.cruise_ship,
    hotel = EXCLUDED.hotel,
    pickup_location = EXCLUDED.pickup_location,
    preferred_language = EXCLUDED.preferred_language,
    booking_stage = EXCLUDED.booking_stage,
    needs_human = EXCLUDED.needs_human,
    updated_at = NOW()
"""


class PostgresConversationRepository(ConversationRepository):
    def get(self, customer_phone: str) -> ConversationState:
        key = normalize_customer_phone(customer_phone)

        with database_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(_SELECT_SQL, (key,))
                row = cursor.fetchone()

        if row is None:
            return ConversationState()
        return db_row_to_state(row)

    def save(self, customer_phone: str, state: ConversationState) -> None:
        key = normalize_customer_phone(customer_phone)
        values = state_to_db_values(key, state)

        with database_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(_UPSERT_SQL, values)
            conn.commit()

    def clear(self) -> None:
        raise NotImplementedError(
            "clear() is not supported by PostgresConversationRepository."
        )
