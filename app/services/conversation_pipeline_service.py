"""Conversation pipeline: deterministic state transition + AI entity extraction.

Coordinates apply_message_to_state → provider.extract_entities →
merge_extraction_into_state. Provider must be injected; provider errors
propagate unchanged; the incoming state is never mutated.
"""

from app.models.conversation import ConversationIntent, ConversationState
from app.services.ai.base import AIProvider
from app.services.conversation_state_service import apply_message_to_state
from app.services.entity_merge_service import merge_extraction_into_state

# Intents for which AI entity extraction is skipped entirely.
_EXTRACTION_SKIPPED_INTENTS: frozenset[ConversationIntent] = frozenset(
    {
        ConversationIntent.GREETING,
        ConversationIntent.HUMAN_REQUEST,
        ConversationIntent.COMPLAINT,
        ConversationIntent.CANCELLATION_REQUEST,
    }
)


class ConversationPipelineService:
    def __init__(self, provider: AIProvider):
        self._provider = provider

    async def process_message(
        self,
        state: ConversationState,
        message: str,
    ) -> ConversationState:
        # 1. Deterministic intent classification + state transition.
        intermediate = apply_message_to_state(state, message)

        # 2. Skip AI extraction where it cannot help.
        if intermediate.intent in _EXTRACTION_SKIPPED_INTENTS:
            return intermediate

        # 3. Structured extraction. AIProviderError propagates unchanged;
        # no partially merged state is returned on failure.
        extraction = await self._provider.extract_entities(message)

        # 4. Safe merge into the intermediate state.
        return merge_extraction_into_state(intermediate, extraction)
