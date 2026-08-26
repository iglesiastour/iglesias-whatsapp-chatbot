"""Tests for the repository selection factory."""

import os
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.postgres_conversation_repository import (
    PostgresConversationRepository,
)
from app.repositories.provider import (
    RepositoryConfigurationError,
    get_conversation_repository,
    get_handoff_repository,
)
from app.services.conversation_store import InMemoryConversationStore
from app.repositories.handoff_repository import HandoffRepository
from app.repositories.in_memory_handoff_repository import InMemoryHandoffRepository
from app.repositories.postgres_handoff_repository import PostgresHandoffRepository


def test_default_backend_is_memory() -> None:
    # The declared field default is memory, independent of ambient env vars.
    from app.config import Settings

    assert Settings.model_fields["conversation_repository_backend"].default == "memory"


def test_memory_backend_returns_conversation_repository() -> None:
    repository = get_conversation_repository()
    assert isinstance(repository, ConversationRepository)


def test_memory_backend_returns_singleton_implementation(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
    assert get_conversation_repository() is get_conversation_repository()
    assert isinstance(get_conversation_repository(), InMemoryConversationStore)


def test_postgres_backend_returns_postgres_repository(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "postgres")
    assert isinstance(
        get_conversation_repository(), PostgresConversationRepository
    )


def test_postgres_factory_does_not_connect_immediately() -> None:
    from unittest.mock import MagicMock as _MagicMock

    fake_connect = _MagicMock()
    with patch(
        "app.db.connection.psycopg.connect", fake_connect
    ):
        with patch.object(
            settings, "conversation_repository_backend", "postgres"
        ):
            repository = get_conversation_repository()

    assert isinstance(repository, PostgresConversationRepository)
    assert fake_connect.call_count == 0


@pytest.mark.parametrize("bad_backend", ["redis", "sqlite", "mysql", "hacker"])
def test_invalid_backend_raises_configuration_error(bad_backend: str) -> None:
    with patch.object(settings, "conversation_repository_backend", bad_backend):
        with pytest.raises(RepositoryConfigurationError):
            get_conversation_repository()


def test_backend_comparison_is_whitespace_normalized(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "  memory  ")
    assert isinstance(get_conversation_repository(), InMemoryConversationStore)

    monkeypatch.setattr(settings, "conversation_repository_backend", "\tpostgres\n")
    assert isinstance(get_conversation_repository(), PostgresConversationRepository)


def test_backend_comparison_is_case_insensitive(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "MEMORY")
    assert isinstance(get_conversation_repository(), InMemoryConversationStore)

    monkeypatch.setattr(settings, "conversation_repository_backend", "Postgres")
    assert isinstance(get_conversation_repository(), PostgresConversationRepository)


def test_error_does_not_echo_raw_configured_value() -> None:
    bad = "super-secret-backend-name"
    with patch.object(settings, "conversation_repository_backend", bad):
        with pytest.raises(RepositoryConfigurationError) as exc_info:
            get_conversation_repository()

    assert bad not in str(exc_info.value)


def test_no_environment_dependency(monkeypatch) -> None:
    snapshot = dict(os.environ)
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
    get_conversation_repository()
    assert dict(os.environ) == snapshot


# --- Handoff repository selection ---


def test_handoff_memory_backend_selection(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
    repository = get_handoff_repository()
    assert isinstance(repository, HandoffRepository)
    assert isinstance(repository, InMemoryHandoffRepository)


def test_handoff_postgres_backend_selection(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "postgres")
    repository = get_handoff_repository()
    assert isinstance(repository, PostgresHandoffRepository)


def test_handoff_postgres_factory_does_not_connect_immediately(monkeypatch) -> None:
    fake_connect = MagicMock()
    with patch("app.db.connection.psycopg.connect", fake_connect):
        with patch.object(settings, "conversation_repository_backend", "postgres"):
            repository = get_handoff_repository()

    assert isinstance(repository, PostgresHandoffRepository)
    assert fake_connect.call_count == 0


@pytest.mark.parametrize("bad_backend", ["redis", "sqlite", "mysql", "hacker"])
def test_handoff_invalid_backend_raises_configuration_error(bad_backend: str) -> None:
    with patch.object(settings, "conversation_repository_backend", bad_backend):
        with pytest.raises(RepositoryConfigurationError):
            get_handoff_repository()


def test_conversation_repository_factory_behavior_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
    assert isinstance(get_conversation_repository(), InMemoryConversationStore)
    monkeypatch.setattr(settings, "conversation_repository_backend", "postgres")
    assert isinstance(get_conversation_repository(), PostgresConversationRepository)


def test_handoff_no_environment_dependency(monkeypatch) -> None:
    snapshot = dict(os.environ)
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
    get_handoff_repository()
    assert dict(os.environ) == snapshot
