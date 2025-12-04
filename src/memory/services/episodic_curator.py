"""Episodic Memory Curator for intelligent conversation storage.

This module uses LLM to determine what conversations are worth storing in episodic memory
and how to best summarize/embed them for future retrieval.
"""

import json
import logging
from typing import Any, Optional, Dict

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
        reasoning: str = ""
    ):
        self.should_store = should_store
        self.content_to_store = content_to_store
        self.summary = summary
        self.importance_score = importance_score
        self.tags = tags or []
        self.reasoning = reasoning


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
        existing_context: str = None
    ) -> EpisodicCurationResult:
        """Use LLM to decide if and how to store an interaction in episodic memory.
        
        Args:
            user_message: User's message content
            assistant_response: Assistant's response content
            user_id: User identifier
            existing_context: Optional context about user's existing memories
            
        Returns:
            Curation result with storage decision and optimized content
        """
        if not settings.openai_api_key:
            # Fallback: store everything with basic rules
            return self._rule_based_curation(user_message, assistant_response)
        
        try:
            # LLM-based curation prompt
            curation_prompt = f"""
You are an episodic memory curator. Analyze this conversation turn and decide if it's worth storing for future reference.

CONVERSATION:
User: {user_message}
Assistant: {assistant_response}

EXISTING CONTEXT: {existing_context or "No prior context"}

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
   - Creating a concise but complete summary
   - Including relevant context and outcomes
   - Adding semantic tags for better searchability

Return JSON format:
{{
    "should_store": true/false,
    "content_to_store": "optimized content for embedding (if storing)",
    "summary": "concise summary for display",
    "importance_score": 0.0-1.0,
    "tags": ["tag1", "tag2", "tag3"],
    "reasoning": "brief explanation of decision"
}}

Focus on preserving information that would be valuable for maintaining context in future conversations.
"""
            
            # Use the summarizer's OpenAI client for consistency
            response = await self.summarizer._call_openai(curation_prompt, max_tokens=400)
            
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
                    return self._rule_based_curation(user_message, assistant_response)
            
            return EpisodicCurationResult(
                should_store=data.get("should_store", True),
                content_to_store=data.get("content_to_store", f"User: {user_message}\nAssistant: {assistant_response}"),
                summary=data.get("summary", ""),
                importance_score=float(data.get("importance_score", 0.5)),
                tags=data.get("tags", []),
                reasoning=data.get("reasoning", "")
            )
            
        except Exception as e:
            logger.error(f"LLM curation failed: {e}, using rule-based fallback")
            return self._rule_based_curation(user_message, assistant_response)
    
    def _rule_based_curation(self, user_message: str, assistant_response: str) -> EpisodicCurationResult:
        """Fallback rule-based curation when LLM is unavailable."""
        user_lower = user_message.lower()
        
        # Simple rules for what NOT to store
        skip_patterns = [
            "hello", "hi", "hey", "good morning", "good evening",
            "thanks", "thank you", "bye", "goodbye",
            "what time is it", "what's the weather",
            "test", "testing", "can you hear me"
        ]
        
        # Check if message is too generic/simple
        if len(user_message.strip()) < 5 or any(pattern in user_lower for pattern in skip_patterns):
            return EpisodicCurationResult(
                should_store=False,
                reasoning="Rule-based: Message too simple or generic"
            )
        
        # Simple rules for what TO store
        store_patterns = [
            "my", "i am", "i like", "i don't like", "i have", "i need",
            "remember", "remind me", "important", "favorite",
            "family", "work", "job", "project"
        ]
        
        importance_score = 0.3  # Default low importance
        if any(pattern in user_lower for pattern in store_patterns):
            importance_score = 0.7  # Higher importance for personal info
        
        return EpisodicCurationResult(
            should_store=True,
            content_to_store=f"User: {user_message}\nAssistant: {assistant_response}",
            summary=user_message[:100] + "..." if len(user_message) > 100 else user_message,
            importance_score=importance_score,
            tags=["personal"] if importance_score > 0.5 else [],
            reasoning="Rule-based: Contains potentially valuable information"
        )


# Factory function
async def create_episodic_curator() -> EpisodicMemoryCurator:
    """Create episodic memory curator."""
    return EpisodicMemoryCurator()