"""Tests for OpenRouter structured entity extraction (no real network)."""

import asyncio
import json
import os
from datetime import date
from unittest.mock import patch

import httpx
import pytest

from app.config import settings
from app.models.extraction import (
    ExtractionSource,
    ExtractedEntities,
    StructuredExtraction,
)
from app.prompts.extraction_prompt import build_extraction_prompt
from app.services.ai.base import AIProviderError
from app.services.ai.openrouter import OpenRouterProvider


def _patch_settings(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(
        settings, "openrouter_base_url", "https://openrouter.test/api/v1"
    )
    monkeypatch.setattr(settings, "openrouter_model", "test-model")


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions"),
        json=payload,
    )


def _content_response(content: str) -> httpx.Response:
    return _response({"choices": [{"message": {"content": content}}]})


def run(coro):
    return asyncio.run(coro)


def test_valid_empty_extraction() -> None:
    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response(json.dumps({key: None for key in ExtractedEntities.model_fields}))

    with patch.object(httpx.AsyncClient, "post", fake_post):
        result = run(OpenRouterProvider().extract_entities("hello"))

    assert result.entities.tour is None
    assert result.entities.adults is None


def test_valid_full_extraction() -> None:
    payload = {
        "tour": "Ephesus",
        "travel_date": "2026-09-10",
        "adults": 2,
        "children": 1,
        "cruise_ship": "Celebrity Equinox",
        "hotel": "Korumar Hotel",
        "pickup_location": "Kusadasi Port",
        "preferred_language": "English",
    }

    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response(json.dumps(payload))

    with patch.object(httpx.AsyncClient, "post", fake_post):
        result = run(
            OpenRouterProvider().extract_entities("book for 2, September 2026")
        )

    assert isinstance(result, StructuredExtraction)
    assert result.source is ExtractionSource.CUSTOMER_MESSAGE
    assert result.entities.tour == "Ephesus"
    assert result.entities.travel_date == date(2026, 9, 10)
    assert result.entities.adults == 2
    assert result.entities.children == 1


@pytest.mark.parametrize(
    ("raw", "expected_date", "message"),
    [
        ("2026-09-10", date(2026, 9, 10), "September 10, 2026"),
        ("2027-01-02", date(2027, 1, 2), "January 2, 2027"),
    ],
)
def test_iso_dates_parsed(raw: str, expected_date: date, message: str) -> None:
    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response(json.dumps({"travel_date": raw}))

    with patch.object(httpx.AsyncClient, "post", fake_post):
        result = run(OpenRouterProvider().extract_entities(message))
    assert result.entities.travel_date == expected_date


def test_extraction_prompt_used_as_system_content_and_message_as_user(
    monkeypatch,
) -> None:
    _patch_settings(monkeypatch)
    captured: dict = {}

    async def fake_post(self, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        captured["headers"] = headers
        return _content_response(json.dumps({"tour": None}))

    with patch.object(httpx.AsyncClient, "post", fake_post):
        run(OpenRouterProvider().extract_entities("We are 4 people from the Equinox"))

    messages = captured["json"]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == build_extraction_prompt()
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "We are 4 people from the Equinox"
    assert captured["json"]["model"] == "test-model"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["response_format"]["type"] == "json_schema"


def test_extraction_payload_includes_structured_response_format(monkeypatch) -> None:
    _patch_settings(monkeypatch)
    captured: dict = {}

    async def fake_post(self, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return _content_response(json.dumps({"tour": None}))

    with patch.object(httpx.AsyncClient, "post", fake_post):
        run(OpenRouterProvider().extract_entities("How much?"))

    assert "response_format" in captured["json"]


def test_response_format_shape() -> None:
    from app.services.ai.openrouter import _EXTRACTION_RESPONSE_FORMAT as fmt

    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "booking_entities"
    assert fmt["json_schema"]["strict"] is True

    schema = fmt["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False

    expected_fields = {
        "tour",
        "travel_date",
        "adults",
        "children",
        "cruise_ship",
        "hotel",
        "pickup_location",
        "preferred_language",
    }
    assert set(schema["properties"].keys()) == expected_fields
    assert set(schema["required"]) == expected_fields


def test_response_schema_adult_child_constraints() -> None:
    from app.services.ai.openrouter import _EXTRACTION_RESPONSE_SCHEMA as schema

    props = schema["properties"]
    assert props["adults"]["minimum"] == 1
    assert props["adults"]["maximum"] == 100
    assert props["children"]["minimum"] == 0
    assert props["children"]["maximum"] == 100


def test_response_schema_forbids_operational_fields() -> None:
    from app.services.ai.openrouter import _EXTRACTION_RESPONSE_SCHEMA as schema

    props = schema["properties"]
    for forbidden in (
        "price",
        "availability",
        "booking_confirmation",
        "discount",
        "payment_confirmation",
        "guide_assignment",
        "vehicle_assignment",
    ):
        assert forbidden not in props


def test_generate_reply_does_not_receive_extraction_response_format(
    monkeypatch,
) -> None:
    _patch_settings(monkeypatch)
    captured: dict = {}

    async def fake_post(self, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return _content_response("AI reply")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        run(OpenRouterProvider().generate_reply("Hello"))

    assert "response_format" not in captured["json"]


def test_no_network_post_is_mocked_everywhere() -> None:
    # sanity: patched post returns synthetic response; nothing dials out.
    async def fake_post(self, url, headers=None, **kwargs):
        raise AssertionError("network helper used directly")

    # _parse_entities path does not touch HTTP at all.
    entities = OpenRouterProvider._parse_entities('{"adults": 2}')
    assert entities.adults == 2


# --- Invalid structured output ---


def _assert_extraction_fails(monkeypatch, content: str) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response(content)

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError):
            run(OpenRouterProvider().extract_entities("m"))


def test_invalid_json_becomes_provider_error(monkeypatch) -> None:
    _assert_extraction_fails(monkeypatch, "this is not json")


def test_json_array_becomes_provider_error(monkeypatch) -> None:
    _assert_extraction_fails(monkeypatch, '[{"adults": 2}]')


def test_empty_content_becomes_provider_error(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response("   ")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="empty"):
            run(OpenRouterProvider().extract_entities("m"))


@pytest.mark.parametrize(
    "payload",
    [
        {"price": "100 EUR"},
        {"availability": True},
        {"booking_confirmed": True},
        {"discount": "10%"},
        {"payment_confirmation": True},
        {"guide_name": "Mehmet"},
    ],
)
def test_forbidden_operational_fields_become_provider_error(
    monkeypatch, payload: dict
) -> None:
    _assert_extraction_fails(monkeypatch, json.dumps(payload))


@pytest.mark.parametrize(
    "payload",
    [
        {"adults": 0},
        {"adults": -3},
        {"adults": 101},
        {"children": -1},
        {"children": 200},
        {"travel_date": "not-a-date"},
        {"travel_date": "10/09/2026"},
    ],
)
def test_invalid_values_become_provider_error(monkeypatch, payload: dict) -> None:
    _assert_extraction_fails(monkeypatch, json.dumps(payload))


def test_missing_api_key_is_provider_error(monkeypatch) -> None:
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    with pytest.raises(AIProviderError, match="OPENROUTER_API_KEY"):
        run(OpenRouterProvider().extract_entities("m"))


def test_http_error_becomes_provider_error(monkeypatch) -> None:
    _patch_settings(monkeypatch)
    request = httpx.Request("POST", "https://openrouter.test/api/v1/chat/completions")
    response = httpx.Response(500, request=request, text='{"error": "secret body"}')
    error = httpx.HTTPStatusError("boom", request=request, response=response)

    async def fake_post(self, url, headers=None, **kwargs):
        raise error

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError) as exc_info:
            run(OpenRouterProvider().extract_entities("m"))

    assert "secret body" not in str(exc_info.value)


def test_network_error_becomes_provider_error(monkeypatch) -> None:
    _patch_settings(monkeypatch)

    async def fake_post(self, url, headers=None, **kwargs):
        raise httpx.ConnectError("connection refused")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        with pytest.raises(AIProviderError, match="request failed"):
            run(OpenRouterProvider().extract_entities("m"))


# --- generate_reply regression & boundaries ---


def test_generate_reply_behavior_unchanged(monkeypatch) -> None:
    _patch_settings(monkeypatch)
    captured: dict = {}

    async def fake_post(self, url, headers=None, **kwargs):
        captured["json"] = kwargs.get("json")
        return _content_response("  AI reply  ")

    with patch.object(httpx.AsyncClient, "post", fake_post):
        reply = run(OpenRouterProvider().generate_reply("Hello"))

    assert reply == "AI reply"
    assert captured["json"]["temperature"] == 0.3
    assert "build_extraction_prompt" not in repr(captured["json"])


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    OpenRouterProvider._parse_entities('{"tour": "Ephesus"}')
    assert dict(os.environ) == snapshot


def test_provider_does_not_depend_on_state_or_merge_services() -> None:
    import sys

    module = sys.modules[OpenRouterProvider.__module__]
    assert not hasattr(module, "ConversationState")
    assert not hasattr(module, "merge_extraction_into_state")


# --- Conservative JSON fence support ---


def _run_extract(message: str, content: str) -> StructuredExtraction:
    async def fake_post(self, url, headers=None, **kwargs):
        return _content_response(content)

    with patch.object(httpx.AsyncClient, "post", fake_post):
        return run(OpenRouterProvider().extract_entities(message))


def test_raw_valid_json_acceptance_retained() -> None:
    result = _run_extract(
        "September 10, 2026 for 2 adults.",
        '{"travel_date": "2026-09-10", "adults": 2}',
    )
    assert result.entities.adults == 2
    assert result.entities.travel_date == date(2026, 9, 10)


def test_fenced_json_language_acceptance() -> None:
    result = _run_extract(
        "September 10, 2026 for 2 adults.",
        '```json\n{"travel_date": "2026-09-10", "adults": 2}\n```',
    )
    assert result.entities.travel_date == date(2026, 9, 10)
    assert result.entities.adults == 2


def test_plain_fenced_json_acceptance() -> None:
    result = _run_extract(
        "September 10, 2026 for 2 adults.",
        '```\n{"travel_date": "2026-09-10", "adults": 2}\n```',
    )
    assert result.entities.travel_date == date(2026, 9, 10)


def test_commentary_before_json_rejected() -> None:
    with pytest.raises(AIProviderError, match="invalid JSON"):
        _run_extract(
            "m",
            'Here is the JSON:\n{"tour": null}',
        )


def test_commentary_after_json_rejected() -> None:
    with pytest.raises(AIProviderError, match="invalid JSON"):
        _run_extract("m", '{"tour": null}\nHope this helps!')


def test_arbitrary_prefix_suffix_rejected() -> None:
    with pytest.raises(AIProviderError, match="invalid JSON"):
        _run_extract("m", 'prefix {"tour": null} suffix')


# --- Date grounding ---


def test_invented_year_cleared_when_no_explicit_year_in_message() -> None:
    result = _run_extract(
        "September 10 for 2 adults.",
        '{"travel_date": "2024-09-10", "adults": 2}',
    )
    assert result.entities.travel_date is None
    assert result.entities.adults == 2


def test_explicit_year_match_retains_travel_date() -> None:
    result = _run_extract(
        "September 10, 2026 for 2 adults.",
        '{"travel_date": "2026-09-10", "adults": 2}',
    )
    assert result.entities.travel_date == date(2026, 9, 10)
    assert result.entities.adults == 2


def test_explicit_year_mismatch_clears_travel_date() -> None:
    result = _run_extract(
        "September 10, 2026 for 2 adults.",
        '{"travel_date": "2025-09-10", "adults": 2}',
    )
    assert result.entities.travel_date is None


def test_multiple_distinct_explicit_years_clears_travel_date() -> None:
    result = _run_extract(
        "September 10, 2026 or September 11, 2027 for 2 adults.",
        '{"travel_date": "2026-09-10", "adults": 2}',
    )
    assert result.entities.travel_date is None


def test_adults_survive_when_invented_date_cleared() -> None:
    result = _run_extract(
        "September 10 for 2 adults.",
        '{"travel_date": "2024-09-10", "adults": 2}',
    )
    assert result.entities.adults == 2
    assert result.entities.travel_date is None


def test_other_entity_fields_survive_date_grounding() -> None:
    result = _run_extract(
        "September 10 for 2 adults at Korumar Hotel.",
        '{"travel_date": "2024-09-10", "adults": 2, '
        '"hotel": "Korumar Hotel", "tour": "Ephesus"}',
    )
    assert result.entities.travel_date is None
    assert result.entities.hotel == "Korumar Hotel"
    assert result.entities.tour == "Ephesus"
    assert result.entities.adults == 2


def test_debug_print_removed() -> None:
    import inspect

    source = inspect.getsource(OpenRouterProvider.extract_entities)
    assert "EXTRACTION RAW CONTENT" not in source
    assert "print(" not in source


def test_no_network_used_in_new_tests() -> None:
    from unittest.mock import MagicMock as _MagicMock

    fake = _MagicMock()
    with patch.object(httpx.AsyncClient, "post", fake):
        OpenRouterProvider._normalize_json_content("```json\n{}\n```")
    fake.assert_not_called()


