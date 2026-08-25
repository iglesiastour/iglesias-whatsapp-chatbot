"""Deterministic tests for the structured extraction models."""

import os
from datetime import date

import pytest
from pydantic import ValidationError

from app.models.extraction import ExtractionSource, ExtractedEntities, StructuredExtraction


def test_empty_extraction_is_valid() -> None:
    extraction = ExtractedEntities()
    assert extraction.tour is None
    assert extraction.travel_date is None
    assert extraction.adults is None
    assert extraction.children is None


def test_all_supported_fields_valid() -> None:
    entities = ExtractedEntities(
        tour="Ephesus",
        travel_date="2026-09-10",
        adults=2,
        children=1,
        cruise_ship="Celebrity Equinox",
        hotel="Korumar Hotel",
        pickup_location="Hotel lobby",
        preferred_language="English",
    )
    assert entities.tour == "Ephesus"
    assert entities.travel_date == date(2026, 9, 10)
    assert entities.adults == 2
    assert entities.children == 1
    assert entities.cruise_ship == "Celebrity Equinox"
    assert entities.hotel == "Korumar Hotel"


def test_iso_date_parsing() -> None:
    assert ExtractedEntities(travel_date="2026-12-31").travel_date == date(2026, 12, 31)


@pytest.mark.parametrize("adults", [1, 100])
def test_adults_bounds_valid(adults: int) -> None:
    assert ExtractedEntities(adults=adults).adults == adults


@pytest.mark.parametrize("adults", [0, -1, 101])
def test_invalid_adults_rejected(adults: int) -> None:
    with pytest.raises(ValidationError):
        ExtractedEntities(adults=adults)


@pytest.mark.parametrize("children", [0, 100])
def test_children_bounds_valid(children: int) -> None:
    assert ExtractedEntities(children=children).children == children


@pytest.mark.parametrize("children", [-1, 101])
def test_invalid_children_rejected(children: int) -> None:
    with pytest.raises(ValidationError):
        ExtractedEntities(children=children)


def test_extraction_source_defaults_to_customer_message() -> None:
    extraction = StructuredExtraction(entities=ExtractedEntities())
    assert extraction.source is ExtractionSource.CUSTOMER_MESSAGE


def test_all_extraction_source_values() -> None:
    assert {source.value for source in ExtractionSource} == {
        "customer_message",
        "human",
        "system",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("price", "100 EUR"),
        ("availability", True),
        ("booking_confirmed", True),
        ("discount", "10%"),
        ("payment_confirmation", True),
        ("guide_name", "Mehmet"),
        ("vehicle_assignment", "Van 1"),
    ],
)
def test_operational_fields_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ExtractedEntities(**{field: value})


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        ExtractedEntities(completely_unknown="x")


def test_no_environment_dependency() -> None:
    snapshot = dict(os.environ)
    ExtractedEntities(tour="Ephesus")
    assert dict(os.environ) == snapshot
