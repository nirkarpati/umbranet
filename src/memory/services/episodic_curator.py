"""Episodic Memory Curator for intelligent conversation storage.

This module uses LLM to determine what conversations are worth storing in 
episodic memory and how to best summarize/embed them for future retrieval.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from ...core.config import settings
from .summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class EpisodicCurationResult:
    """Result of episodic memory curation."""
    
    def __init__(
        self, 
        should_store: bool, 
        content_to_store: str = "", 
        summary: str = "", 
        importance_score: float = 0.0,
        tags: list[str] = None,
        reasoning: str = "",
        occurred_at: str | None = None
    ):
        self.should_store = should_store
        self.content_to_store = content_to_store
        self.summary = summary
        self.importance_score = importance_score
        self.tags = tags or []
        self.reasoning = reasoning
        self.occurred_at = occurred_at


class EpisodicMemoryCurator:
    """LLM-powered curator for deciding what to store in episodic memory."""
    
    def __init__(self):
        """Initialize episodic memory curator."""
        self.summarizer = ConversationSummarizer()
    
    async def __aenter__(self) -> "EpisodicMemoryCurator":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass
    
    async def curate_interaction(
        self, 
        user_message: str, 
        assistant_response: str,
        user_id: str,
        existing_context: str = None,
        current_time: datetime = None
    ) -> EpisodicCurationResult:
        """Use LLM to decide if and how to store an interaction in episodic memory.
        
        Args:
            user_message: User's message content
            assistant_response: Assistant's response content
            user_id: User identifier
            existing_context: Optional context about user's existing memories
            current_time: Current datetime for temporal grounding
            
        Returns:
            Curation result with storage decision and optimized content
        """
        if not settings.openai_api_key:
            logger.warning(
                "OpenAI API key not configured - cannot perform LLM curation"
            )
            return EpisodicCurationResult(
                should_store=False,
                reasoning="OpenAI API key not configured"
            )
        
        # Set current time for temporal grounding
        if current_time is None:
            current_time = datetime.utcnow()
        
        try:
            # LLM-based curation prompt with temporal grounding
            curation_prompt = f"""
You are an episodic memory curator. Analyze this conversation turn and decide 
if it's worth storing for future reference.

CONVERSATION:
User: {user_message}
Assistant: {assistant_response}

EXISTING CONTEXT: {existing_context or "No prior context"}

TEMPORAL GROUNDING - CRITICAL:
Reference Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC

Instructions:
1. Rewrite relative time expressions ('yesterday', 'last month', 
   'a few days ago', etc.) in content_to_store to absolute dates based 
   on Reference Time.
2. If a specific event date is identified, extract it to the occurred_at 
   field (ISO 8601 YYYY-MM-DD format).

Examples:
- "I went surfing yesterday" → "User went surfing on 
  {(current_time - timedelta(days=1)).strftime('%Y-%m-%d')}"
- "Last month I visited Paris" → "User visited Paris in 
  {(current_time.replace(day=1) - timedelta(days=1)).strftime('%B %Y')}"

CURATION CRITERIA:
1. Store conversations that contain:
   - New factual information about the user (personal details, preferences, experiences)
   - Important decisions, goals, or commitments
   - Significant events, stories, or experiences shared
   - Learning moments or insights
   - Problem-solving discussions with concrete outcomes
   - Complex multi-turn conversations with context

2. DON'T store conversations that are:
   - Simple greetings or small talk
   - Generic information requests (weather, facts, definitions)
   - Repetitive or redundant information
   - Test messages or debugging
   - System status checks

3. If storing, optimize the content for future retrieval by:
   - Converting ALL relative time references to absolute dates
   - Creating a concise but complete summary
   - Including relevant context and outcomes
   - Adding semantic tags for better searchability

Return JSON format:
{{
    "should_store": true/false,
    "content_to_store": "optimized content with absolute dates (if storing)",
    "summary": "concise summary for display",
    "importance_score": 0.0-1.0,
    "tags": ["tag1", "tag2", "tag3"],
    "reasoning": "brief explanation of decision",
    "occurred_at": "YYYY-MM-DD if specific event date identified, null otherwise"
}}

Focus on preserving information with accurate temporal context for future conversations.
"""
            
            # Use the summarizer's OpenAI client for consistency
            response = await self.summarizer._call_openai(
                curation_prompt, max_tokens=500
            )
            
            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    logger.error("Could not parse LLM curation response as JSON")
                    return EpisodicCurationResult(
                        should_store=False,
                        reasoning="Failed to parse LLM response"
                    )
            
            return EpisodicCurationResult(
                should_store=data.get("should_store", False),
                content_to_store=data.get(
                    "content_to_store", 
                    f"User: {user_message}\nAssistant: {assistant_response}"
                ),
                summary=data.get("summary", ""),
                importance_score=float(data.get("importance_score", 0.5)),
                tags=data.get("tags", []),
                reasoning=data.get("reasoning", ""),
                occurred_at=data.get("occurred_at")
            )
            
        except Exception as e:
            logger.error(f"LLM curation failed: {e}")
            return EpisodicCurationResult(
                should_store=False,
                reasoning=f"LLM curation error: {str(e)}"
            )


# Factory function
async def create_episodic_curator() -> EpisodicMemoryCurator:
    """Create episodic memory curator."""
    return EpisodicMemoryCurator()