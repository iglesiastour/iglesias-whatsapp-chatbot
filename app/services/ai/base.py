"""Abstract AI provider contract."""

from abc import ABC, abstractmethod

from app.models.extraction import StructuredExtraction


class AIProviderError(Exception):
    """Base error raised by AI providers."""


class AIProvider(ABC):
    @abstractmethod
    async def generate_reply(
        self,
        message: str,
        conversation_context: str | None = None,
    ) -> str:
        """Generate a reply for a customer message."""
        raise NotImplementedError

    @abstractmethod
    async def extract_entities(self, message: str) -> StructuredExtraction:
        """Extract structured booking entities from a customer message."""
        raise NotImplementedError