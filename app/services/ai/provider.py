"""Factory for the active AI provider."""

from app.services.ai.base import AIProvider
from app.services.ai.openrouter import OpenRouterProvider


def get_ai_provider() -> AIProvider:
    """Return the configured AI provider."""
    return OpenRouterProvider()
