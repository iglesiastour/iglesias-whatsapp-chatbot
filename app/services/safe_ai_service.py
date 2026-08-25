"""Safety-orchestrated AI service.

Coordinates the prompt guard, the AI provider, and the output guard into a
single safe reply pipeline. Provider errors are intentionally propagated
unchanged — the API route owns provider-error → HTTP behavior.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.models.conversation import BookingStage
from app.prompts.policies import get_safety_fallback
from app.security.output_guard import inspect_ai_output
from app.security.prompt_guard import inspect_prompt
from app.services.ai.base import AIProvider
from app.services.safety_fallback_service import (
    SafetyFallbackContext,
    build_contextual_safety_fallback,
)

INPUT_SAFETY_REPLY = (
    "I can help with your tour, booking, itinerary, or travel questions."
)


class SafeAIOutcome(StrEnum):
    GENERATED = "generated"
    INPUT_BLOCKED = "input_blocked"
    OUTPUT_BLOCKED = "output_blocked"


@dataclass(frozen=True)
class SafeAIResult:
    reply: str
    outcome: SafeAIOutcome


class SafeAIService:
    def __init__(self, provider: AIProvider):
        self._provider = provider

    async def generate_reply(
        self,
        message: str,
        conversation_context: str | None = None,
        known_tour: str | None = None,
        booking_stage: BookingStage | None = None,
        fallback_context: SafetyFallbackContext | None = None,
    ) -> SafeAIResult:
        # 1. Input guard: never send manipulative messages to the provider.
        input_result = inspect_prompt(message)
        if not input_result.is_safe:
            return SafeAIResult(
                reply=INPUT_SAFETY_REPLY,
                outcome=SafeAIOutcome.INPUT_BLOCKED,
            )

        # 2. Provider call. AIProviderError propagates unchanged by design.
        reply = await self._provider.generate_reply(
            message,
            conversation_context=conversation_context,
        )

        # 3. Output guard: replace unsafe replies with the policy fallback
        # for the first (deterministically ordered) violation.
        output_result = inspect_ai_output(
            reply,
            known_tour=known_tour,
            booking_stage=booking_stage,
        )
        if not output_result.is_safe:
            category = output_result.violations[0]
            return SafeAIResult(
                reply=build_contextual_safety_fallback(category, fallback_context),
                outcome=SafeAIOutcome.OUTPUT_BLOCKED,
            )

        return SafeAIResult(reply=reply, outcome=SafeAIOutcome.GENERATED)
