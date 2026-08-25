"""Shared pytest configuration.

Safety guard: automated tests must never touch a real PostgreSQL/Neon
instance. The in-memory backend is forced unless a specific test explicitly
overrides it (e.g., provider-selection tests patching this same setting).
"""

import pytest

from app.config import settings


@pytest.fixture(autouse=True)
def force_in_memory_repository_backend(monkeypatch):
    monkeypatch.setattr(settings, "conversation_repository_backend", "memory")
