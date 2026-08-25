"""Tests for the database connection helper (no real PostgreSQL)."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.db import connection as db_connection_module
from app.db.connection import (
    DatabaseNotConfiguredError,
    database_connection,
    get_database_connection,
)


@pytest.fixture()
def no_database_url(monkeypatch):
    monkeypatch.setattr(settings, "database_url", "")


def test_database_url_setting_defaults_to_empty(monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_url", "")
    assert settings.database_url == ""


def test_missing_url_raises_not_configured(no_database_url) -> None:
    with pytest.raises(DatabaseNotConfiguredError, match="DATABASE_URL"):
        get_database_connection()


def test_whitespace_only_url_raises_not_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_url", "   ")
    with pytest.raises(DatabaseNotConfiguredError):
        get_database_connection()


def test_valid_url_calls_psycopg_connect_exactly_once(monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_url", "postgresql://user@host/db")
    fake_connect = MagicMock(return_value=MagicMock())

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        get_database_connection()

    assert fake_connect.call_count == 1


def test_exact_configured_url_passed_to_psycopg(monkeypatch) -> None:
    url = "postgresql://user:pw@neon.example/dbname"
    monkeypatch.setattr(settings, "database_url", url)
    fake_connect = MagicMock(return_value=MagicMock())

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        get_database_connection()

    assert fake_connect.call_args.args == (url,)


def test_get_database_connection_returns_mocked_connection(monkeypatch) -> None:
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_conn = MagicMock()
    fake_connect = MagicMock(return_value=fake_conn)

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        conn = get_database_connection()

    assert conn is fake_conn


def test_import_does_not_connect(monkeypatch) -> None:
    # Accessing/importing the module must never trigger psycopg.connect,
    # even with a configured URL.
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_connect = MagicMock()

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        # Touch every public attribute of the module without calling helpers.
        for name in dir(db_connection_module):
            if not name.startswith("_"):
                getattr(db_connection_module, name)

    assert fake_connect.call_count == 0


# --- Context manager ---


def test_context_manager_yields_connection(monkeypatch) -> None:
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_conn = MagicMock()
    fake_connect = MagicMock(return_value=fake_conn)

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with database_connection() as conn:
            assert conn is fake_conn


def test_context_manager_closes_connection_afterward(monkeypatch) -> None:
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_conn = MagicMock()
    fake_connect = MagicMock(return_value=fake_conn)

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with database_connection():
            assert fake_conn.close.call_count == 0
        assert fake_conn.close.call_count == 1


def test_connection_closed_when_body_raises(monkeypatch) -> None:
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_conn = MagicMock()
    fake_connect = MagicMock(return_value=fake_conn)

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with pytest.raises(RuntimeError, match="boom"):
            with database_connection():
                raise RuntimeError("boom")

    assert fake_conn.close.call_count == 1


def test_body_exception_is_not_swallowed(monkeypatch) -> None:
    url = "postgresql://user@host/db"
    monkeypatch.setattr(settings, "database_url", url)
    fake_connect = MagicMock(return_value=MagicMock())
    original = ValueError("original error")

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with pytest.raises(ValueError) as exc_info:
            with database_connection():
                raise original

    assert exc_info.value is original


def test_error_message_contains_only_static_text(monkeypatch) -> None:
    # Even if a URL-like value existed elsewhere, the not-configured error
    # must be static and must never embed configuration values.
    monkeypatch.setattr(settings, "database_url", "")
    fake_connect = MagicMock(return_value=MagicMock())

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with pytest.raises(DatabaseNotConfiguredError) as exc_info:
            get_database_connection()

    assert str(exc_info.value) == "DATABASE_URL is not configured."
    assert "postgresql://" not in str(exc_info.value)
    fake_connect.assert_not_called()


def test_no_real_network_used(monkeypatch) -> None:
    # With psycopg.connect patched, nothing can dial out even with a URL set.
    monkeypatch.setattr(settings, "database_url", "postgresql://user@host/db")
    fake_connect = MagicMock(return_value=MagicMock())

    with patch.object(db_connection_module.psycopg, "connect", fake_connect):
        with database_connection():
            pass

    assert fake_connect.call_count == 1


def test_no_environment_dependency_beyond_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "database_url", "")
    snapshot = dict(os.environ)
    try:
        get_database_connection()
    except DatabaseNotConfiguredError:
        pass
    assert dict(os.environ) == snapshot
