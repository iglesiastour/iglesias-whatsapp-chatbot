import httpx

from app.config import settings


class OpenRouterError(Exception):
    """Raised when OpenRouter cannot return a usable AI response."""


async def generate_reply(message: str) -> str:
    """Generate a chatbot reply through OpenRouter."""

    if not settings.openrouter_api_key:
        raise OpenRouterError("OPENROUTER_API_KEY is not configured.")

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
                "content": (
                    "You are the AI sales assistant for Iglesias Tour Turkey. "
                    "Be helpful, professional, concise, and friendly. "
                    "Do not invent tour availability, prices, or booking confirmations."
                ),
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
        print("OPENROUTER STATUS:", exc.response.status_code)
        print("OPENROUTER BODY:", exc.response.text)
        raise OpenRouterError("OpenRouter request failed.") from exc

    except httpx.HTTPError as exc:
        print("OPENROUTER ERROR:", repr(exc))
        raise OpenRouterError("OpenRouter request failed.") from exc

    try:
        data = response.json()
        reply = data["choices"][0]["message"]["content"]

    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise OpenRouterError("OpenRouter returned an invalid response.") from exc

    if not isinstance(reply, str) or not reply.strip():
        raise OpenRouterError("OpenRouter returned an empty response.")

    return reply.strip()