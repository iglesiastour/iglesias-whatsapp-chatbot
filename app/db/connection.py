"""Minimal synchronous PostgreSQL connection helper.

Nothing connects at import time: DATABASE_URL is only used when
get_database_connection()/database_connection() are explicitly called.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg

from app.config import settings


class DatabaseNotConfiguredError(Exception):
    """Raised when DATABASE_URL is not configured."""


def get_database_connection():
    """Open a new psycopg connection from the configured DATABASE_URL."""
    database_url = settings.database_url.strip()
    if not database_url:
        raise DatabaseNotConfiguredError("DATABASE_URL is not configured.")

    return psycopg.connect(database_url)


@contextmanager
def database_connection() -> Iterator:
    """Yield a connection and always close it; transactions stay caller-owned."""
    connection = get_database_connection()
    try:
        yield connection
    finally:
        connection.close()
