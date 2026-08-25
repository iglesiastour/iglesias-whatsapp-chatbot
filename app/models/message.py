from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TestMessageRequest(BaseModel):
    """A simulated inbound WhatsApp message."""

    model_config = ConfigDict(populate_by_name=True)

    sender: Annotated[str, Field(alias="from", min_length=1)]
    name: str | None = None
    message: str = Field(min_length=1)

    @field_validator("sender", "message", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("name", mode="before")
    @classmethod
    def strip_optional_name(cls, value: object) -> object:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value


class NormalizedMessage(BaseModel):
    customer_phone: str
    customer_name: str | None
    message: str
    source: Literal["test"]
    received_at: datetime


class MessageData(BaseModel):
    customer_phone: str
    customer_name: str | None
    message: str
    source: Literal["test"]


class TestMessageResponse(BaseModel):
    success: bool
    data: MessageData


class ProcessMessageData(BaseModel):
    customer_phone: str
    reply: str


class ProcessMessageResponse(BaseModel):
    success: bool
    data: ProcessMessageData
