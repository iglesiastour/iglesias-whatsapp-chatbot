"""OpenRouter implementation of the AI provider interface."""

import json
import re
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

# OpenRouter/OpenAI-compatible structured-output JSON Schema for booking
# entity extraction. Supplemented by, not replacing, build_extraction_prompt().
_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "tour": {"type": ["string", "null"]},
        "travel_date": {
            "type": ["string", "null"],
            "description": (
                "ISO date YYYY-MM-DD only when explicitly grounded in the "
                "customer message"
            ),
        },
        "adults": {"type": ["integer", "null"], "minimum": 1, "maximum": 100},
        "children": {"type": ["integer", "null"], "minimum": 0, "maximum": 100},
        "cruise_ship": {"type": ["string", "null"]},
        "hotel": {"type": ["string", "null"]},
        "pickup_location": {"type": ["string", "null"]},
        "preferred_language": {"type": ["string", "null"]},
    },
    "required": [
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    ],
    "additionalProperties": False,
}

_EXTRACTION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "booking_entities",
        "strict": True,
        "schema": _EXTRACTION_RESPONSE_SCHEMA,
    },
}


class OpenRouterProvider(AIProvider):
    """Generate chatbot replies and structured extractions through OpenRouter."""

    async def generate_reply(
        self,
        message: str,
        conversation_context: str | None = None,
    ) -> str:
        if not settings.openrouter_api_key:
            raise AIProviderError("OPENROUTER_API_KEY is not configured.")

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": build_system_prompt(),
            },
        ]

        if conversation_context and conversation_context.strip():
            messages.append(
                {
                    "role": "system",
                    "content": conversation_context,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": message,
            }
        )

        payload = {
            "model": settings.openrouter_model,
            "messages": messages,
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
            "response_format": _EXTRACTION_RESPONSE_FORMAT,
        }

        response = await self._post_chat_completion(payload)
        content = self._extract_response_content(response)

        entities = self._parse_entities(content)
        entities = self._ground_travel_date(entities, message)

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
        normalized = OpenRouterProvider._normalize_json_content(content)

        try:
            parsed = json.loads(normalized)
        except (ValueError, TypeError) as exc:
            raise AIProviderError("OpenRouter returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise AIProviderError("OpenRouter extraction must be a JSON object.")

        try:
            return ExtractedEntities.model_validate(parsed)
        except ValueError as exc:
            raise AIProviderError("OpenRouter extraction failed validation.") from exc

    @staticmethod
    def _normalize_json_content(content: str) -> str:
        """Strip exactly one complete Markdown code fence around a JSON object.

        Only a full-line triple-backtick fence (optionally with `json`) is
        accepted. Arbitrary prose around JSON is left intact so json.loads
        fails on it.
        """
        stripped = content.strip()
        lines = stripped.splitlines()

        if len(lines) < 3:
            return stripped

        first = lines[0].strip()
        last = lines[-1].strip()

        if first == "```" or first == "```json":
            if last == "```":
                body = "\n".join(lines[1:-1]).strip()
                if body.startswith("{") and body.endswith("}"):
                    return body

        return stripped

    @staticmethod
    def _ground_travel_date(
        entities: ExtractedEntities,
        message: str,
    ) -> ExtractedEntities:
        """Drop extracted travel_date unless the customer (not the model)
        supplied a matching explicit year."""
        travel_date = entities.travel_date
        if travel_date is None:
            return entities

        explicit_years = set(OpenRouterProvider._explicit_years(message))

        if len(explicit_years) != 1:
            # Zero, or multiple distinct, explicit years cannot be trusted
            # deterministically.
            return entities.model_copy(update={"travel_date": None})

        explicit_year = next(iter(explicit_years))
        if travel_date.year != explicit_year:
            return entities.model_copy(update={"travel_date": None})

        return entities

    @staticmethod
    def _explicit_years(message: str) -> list[int]:
        """Return 20xx-like year tokens explicitly present in the message."""
        return [int(m) for m in re.findall(r"\b(20\d\d)\b", message)]
