"""OpenRouter implementation of the AI provider interface."""

import httpx

from app.config import settings
from app.prompts.system_prompt import build_system_prompt
from app.services.ai.base import AIProvider, AIProviderError


class OpenRouterProvider(AIProvider):
    """Generate chatbot replies through the OpenRouter chat completions API."""

    async def generate_reply(self, message: str) -> str:
        if not settings.openrouter_api_key:
            raise AIProviderError("OPENROUTER_API_KEY is not configured.")

        url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
        }

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

        try:
            data = response.json()
            reply = data["choices"][0]["message"]["content"]

        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise AIProviderError("OpenRouter returned an invalid response.") from exc

        if not isinstance(reply, str) or not reply.strip():
            raise AIProviderError("OpenRouter returned an empty response.")

        return reply.strip()
