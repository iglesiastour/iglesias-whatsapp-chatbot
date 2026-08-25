"""Unit tests for the OpenRouter AI provider (no real network calls)."""

import asyncio
from unittest.mock import patch

import httpx
import pytest

from app.config import settings
from app.services.ai.base import AIProvider, AIProviderError
from app.services.ai.openrouter import OpenRouterProvider


def _patch_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "openrouter_base_url", "https://openrouter.test/api/v1")
    monkeypatch.setattr(settings, "openrouter_model", "test-model")


def _run(coro):
    return asyncio.run(coro)


def test_provider_is_an_ai_provider() -> None:
    assert isinstance(OpenRouterProvider(), AIProvider)


def test_generate_reply_uses_centralized_system_prompt(monkeypatch) -> None:
    """The system message must come from build_system_prompt(), not a hardcoded string."""

    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json

        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "AI reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="TEST CENTRAL SYSTEM PROMPT",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        result = _run(OpenRouterProvider().generate_reply("Hello"))

    assert result == "AI reply"
    assert captured["url"] == "https://openrouter.test/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"

    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "TEST CENTRAL SYSTEM PROMPT"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_generate_reply_strips_whitespace(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, json=None):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "  AI reply  "}}]},
        )

    with patch.object(httpx.AsyncClient, "post", fake_post):
        result = _run(OpenRouterProvider().generate_reply("Hello"))

    assert result == "AI reply"


def test_generate_reply_rejects_empty_response(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, json=None):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "   "}}]},
        )

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="empty response"):
            _run(OpenRouterProvider().generate_reply("Hello"))


def test_generate_reply_rejects_invalid_response(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, json=None):
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"unexpected": True},
        )

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="invalid response"):
            _run(OpenRouterProvider().generate_reply("Hello"))


def test_http_status_error_becomes_provider_error(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    request = httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    response = httpx.Response(500, request=request)
    error = httpx.HTTPStatusError("boom", request=request, response=response)

    async def fake_post(self, url, headers=None, json=None):
        raise error

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="request failed"):
            _run(OpenRouterProvider().generate_reply("Hello"))


def test_network_error_becomes_provider_error(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, json=None):
        raise httpx.ConnectError("connection refused")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="request failed"):
            _run(OpenRouterProvider().generate_reply("Hello"))


def test_missing_api_key_is_a_provider_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "")

    with pytest.raises(AIProviderError, match="OPENROUTER_API_KEY"):
        _run(OpenRouterProvider().generate_reply("Hello"))


# --- Context-aware reply tests ---


def test_no_context_keeps_two_message_structure(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(OpenRouterProvider().generate_reply("Hello"))

    messages = captured["json"]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_context_supplied_creates_three_messages(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "Hello",
                conversation_context="Known tour: Ephesus",
            )
        )

    messages = captured["json"]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "SYS"
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] == "Known tour: Ephesus"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Hello"


def test_context_message_is_second_system_message(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "msg",
                conversation_context="ctx",
            )
        )

    messages = captured["json"]["messages"]
    assert messages[0]["content"] == "SYS"
    assert messages[1]["content"] == "ctx"
    assert messages[2]["content"] == "msg"


def test_whitespace_only_context_omitted(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "Hello",
                conversation_context="   ",
            )
        )

    messages = captured["json"]["messages"]
    assert len(messages) == 2


def test_context_does_not_expose_customer_phone(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "Hello",
                conversation_context="Known tour: Ephesus",
            )
        )

    full_payload = str(captured["json"])
    assert "+905551112233" not in full_payload


def test_temperature_unchanged_with_context(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "Hello",
                conversation_context="ctx",
            )
        )

    assert captured["json"]["temperature"] == 0.3


def test_no_response_format_on_normal_reply(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    captured: dict = {}

    async def fake_post(self, url, headers=None, json=None):
        captured["json"] = json
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"message": {"content": "reply"}}]},
        )

    with (
        patch(
            "app.services.ai.openrouter.build_system_prompt",
            return_value="SYS",
        ),
        patch.object(httpx.AsyncClient, "post", fake_post),
    ):
        _run(
            OpenRouterProvider().generate_reply(
                "Hello",
                conversation_context="ctx",
            )
        )

    assert "response_format" not in captured["json"]

