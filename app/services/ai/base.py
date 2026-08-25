"""Abstract AI provider contract."""

from abc import ABC, abstractmethod


class AIProviderError(Exception):
    """Base error raised by AI providers."""


class AIProvider(ABC):
    @abstractmethod
    async def generate_reply(self, message: str) -> str:
        """Generate a reply for a customer message."""
        raise NotImplementedError
