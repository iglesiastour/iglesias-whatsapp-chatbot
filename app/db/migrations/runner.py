"""Explicit, deterministic migration registry and runner.

Migrations are explicit SQL files registered in code (no dynamic file
execution). Running them is idempotent by design and only ever happens when
run_migrations() is explicitly invoked — never at import time.
"""

from dataclasses import dataclass
from pathlib import Path

from app.db.connection import database_connection


@dataclass(frozen=True)
class Migration:
    version: str
    filename: str


# Registered in deterministic execution order.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(version="0001", filename="0001_conversation_states.sql"),
    Migration(version="0002", filename="0002_handoff_requests.sql"),
)

_MIGRATIONS_DIR = Path(__file__).resolve().parent


def load_migration_sql(migration: Migration) -> str:
    """Load the SQL text for a registered migration.

    Only filenames registered in MIGRATIONS may be loaded; arbitrary paths
    are rejected.
    """
    path = _MIGRATIONS_DIR / migration.filename
    if path.parent != _MIGRATIONS_DIR or not path.is_file():
        raise FileNotFoundError(f"Migration file missing: {migration.filename}")

    if migration not in MIGRATIONS:
        raise ValueError(f"Unknown migration: {migration.filename}")

    return path.read_text(encoding="utf-8")


def run_migrations() -> None:
    """Execute all registered idempotent migrations in order."""
    for migration in MIGRATIONS:
        sql = load_migration_sql(migration)
        with database_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
            conn.commit()


if __name__ == "__main__":
    run_migrations()
