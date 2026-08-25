"""OpenRouter implementation of the AI provider interface."""

import json
from typing import Any

import httpx

from app.config import settings
from app.models.extraction import (
    ExtractionSource,
    ExtractedEntities,
    StructuredExtraction,
)
from app.prompts.extraction_prompt import build_extraction_prompt
from app.prompts.system_prompt import build_system_prompt
from app.services.ai.base import AIProvider, AIProviderError


class OpenRouterProvider(AIProvider):
    """Generate chatbot replies and structured extractions through OpenRouter."""

    async def generate_reply(self, message: str) -> str:
        if not settings.openrouter_api_key:
            raise AIProviderError("OPENROUTER_API_KEY is not configured.")

        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": build_system_prompt(),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
            "temperature": 0.3,
        }

        response = await self._post_chat_completion(payload)
        content = self._extract_response_content(response)

        if not content:
            raise AIProviderError("OpenRouter returned an empty response.")

        return content

    async def extract_entities(self, message: str) -> StructuredExtraction:
        if not settings.openrouter_api_key:
            raise AIProviderError("OPENROUTER_API_KEY is not configured.")

        payload = {
            "model": settings.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": build_extraction_prompt(),
                },
                {
                    "role": "user",
                    "content": message,
                },
            ],
        }

        response = await self._post_chat_completion(payload)
        content = self._extract_response_content(response)

        entities = self._parse_entities(content)

        return StructuredExtraction(
            entities=entities,
            source=ExtractionSource.CUSTOMER_MESSAGE,
        )

    # -- internals ---------------------------------------------------------

    async def _post_chat_completion(self, payload: dict[str, Any]) -> httpx.Response:
        """POST a chat-completion request with shared auth/URL/timeout behavior."""
        url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()

        except httpx.HTTPStatusError as exc:
            raise AIProviderError("OpenRouter request failed.") from exc

        except httpx.HTTPError as exc:
            raise AIProviderError("OpenRouter request failed.") from exc

        return response

    @staticmethod
    def _extract_response_content(response: httpx.Response) -> str:
        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIProviderError("OpenRouter returned an invalid response.") from exc

        if not isinstance(content, str):
            raise AIProviderError("OpenRouter returned an invalid response.")
        if not content.strip():
            raise AIProviderError("OpenRouter returned an empty response.")

        return content.strip()

    @staticmethod
    def _parse_entities(content: str) -> ExtractedEntities:
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError) as exc:
            raise AIProviderError("OpenRouter returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise AIProviderError("OpenRouter extraction must be a JSON object.")

        try:
            return ExtractedEntities.model_validate(parsed)
        except ValueError as exc:
            raise AIProviderError("OpenRouter extraction failed validation.") from exc
