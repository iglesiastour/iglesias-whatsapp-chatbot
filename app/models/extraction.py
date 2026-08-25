"""Typed models for booking-related entities extracted from customer messages.

This is the contract future extraction (LLM or rule-based) must obey.
Operational/business facts (price, confirmation, discounts, etc.) are
deliberately NOT representable here.
"""

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ExtractionSource(StrEnum):
    CUSTOMER_MESSAGE = "customer_message"
    HUMAN = "human"
    SYSTEM = "system"


class ExtractedEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tour: str | None = None
    travel_date: date | None = None

    adults: int | None = Field(default=None, ge=1, le=100)
    children: int | None = Field(default=None, ge=0, le=100)

    cruise_ship: str | None = None
    hotel: str | None = None
    pickup_location: str | None = None
    preferred_language: str | None = None


class StructuredExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: ExtractedEntities
    source: ExtractionSource = ExtractionSource.CUSTOMER_MESSAGE
