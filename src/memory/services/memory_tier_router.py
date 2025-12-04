"""Memory Tier Router for intelligent storage decisions.

This module uses LLM to determine which memory tiers are appropriate 
for different types of information.
"""

import json
import logging
from typing import Dict, List, Any
from enum import Enum

from ...core.config import settings
from .summarizer import ConversationSummarizer

logger = logging.getLogger(__name__)


class MemoryTier(str, Enum):
    """Memory tier types for routing decisions."""
    SHORT_TERM = "short_term"      # Always stored - working context
    EPISODIC = "episodic"          # Personal experiences, events, stories
    SEMANTIC = "semantic"          # Facts, entities, relationships, knowledge
    PROCEDURAL = "procedural"      # Preferences, rules, behavioral patterns


class TierRoutingResult:
    """Result of memory tier routing decision."""
    
    def __init__(
        self,
        recommended_tiers: List[MemoryTier],
        reasoning: Dict[str, str],
        content_adaptations: Dict[str, str] = None
    ):
        self.recommended_tiers = recommended_tiers
        self.reasoning = reasoning  # tier -> reason mapping
        self.content_adaptations = content_adaptations or {}  # tier -> adapted content


class MemoryTierRouter:
    """LLM-powered router for deciding which memory tiers to use."""
    
    def __init__(self):
        """Initialize memory tier router."""
        self.summarizer = ConversationSummarizer()
    
    async def __aenter__(self) -> "MemoryTierRouter":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        pass
    
    async def route_interaction(
        self,
        user_message: str,
        assistant_response: str,
        user_id: str
    ) -> TierRoutingResult:
        """Determine which memory tiers should store this interaction.
        
        Args:
            user_message: User's message content
            assistant_response: Assistant's response content
            user_id: User identifier
            
        Returns:
            Routing result with recommended tiers and reasoning
        """
        if not settings.openai_api_key:
            # Fallback: use rule-based routing
            return self._rule_based_routing(user_message, assistant_response)
        
        try:
            routing_prompt = f"""
You are a memory tier router. Analyze this conversation and determine which memory tiers should store this information.

CONVERSATION:
User: {user_message}
Assistant: {assistant_response}

MEMORY TIER OPTIONS:
1. SHORT_TERM: Always stored - working conversation context (don't specify this, it's automatic)
2. EPISODIC: Personal experiences, events, stories, things that happened to the user
3. SEMANTIC: Facts, entities, relationships, knowledge about people/places/things  
4. PROCEDURAL: User preferences, rules, behavioral patterns, likes/dislikes

ROUTING GUIDELINES:

EPISODIC Examples:
- "I went to the beach and found a golden ring" (experience/event)
- "Yesterday I had dinner with my mom" (experience/event)
- "I watched Terminator 3 and it was terrible" (experience + reaction)
- "That movie was awful, I didn't enjoy it" (reaction to experience)
- "I got promoted at work" (life event)
- "I traveled to Paris last summer" (experience)
- Personal stories, experiences, events, AND reactions to experiences

SEMANTIC Examples:
- "My mom's name is Sarah, she's 65" (factual information)
- "I work at Google as a software engineer" (current fact)
- "My dog Benji is a Border Collie" (factual relationship)
- Facts about people, places, things, relationships

PROCEDURAL Examples:
- "I prefer tea over coffee" (general preference, not tied to specific experience)
- "Always remind me about meetings 10 minutes early" (behavioral rule)
- "I don't like spicy food in general" (general preference)
- Preferences, behavioral rules, patterns NOT tied to specific experiences

IMPORTANT DISTINCTION:
- "I watched Movie X and hated it" → EPISODIC (specific experience + reaction)
- "I hate action movies in general" → PROCEDURAL (general preference)
- "The restaurant was great when I went there" → EPISODIC (specific experience)
- "I always prefer Italian food" → PROCEDURAL (general preference)

MIXED Examples:
- "I love the new restaurant downtown" → EPISODIC (if about a visit) + PROCEDURAL (general preference)
- "My sister Jane lives in Boston" → SEMANTIC (fact) + EPISODIC (if mentioned in story context)

Return JSON format:
{{
    "tiers": ["episodic", "semantic", "procedural"],
    "reasoning": {{
        "episodic": "reason if included",
        "semantic": "reason if included", 
        "procedural": "reason if included"
    }}
}}

Focus on choosing the RIGHT tiers, not all tiers. Some conversations might only need one tier.
"""
            
            response = await self.summarizer._call_openai(routing_prompt, max_tokens=300)
            
            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    logger.error("Could not parse LLM routing response as JSON")
                    return self._rule_based_routing(user_message, assistant_response)
            
            # Convert string tier names to enum values
            recommended_tiers = []
            for tier_name in data.get("tiers", []):
                try:
                    tier_enum = MemoryTier(tier_name)
                    recommended_tiers.append(tier_enum)
                except ValueError:
                    logger.warning(f"Invalid tier name from LLM: {tier_name}")
            
            # Always include short-term (it's automatic)
            if MemoryTier.SHORT_TERM not in recommended_tiers:
                recommended_tiers.insert(0, MemoryTier.SHORT_TERM)
            
            return TierRoutingResult(
                recommended_tiers=recommended_tiers,
                reasoning=data.get("reasoning", {})
            )
            
        except Exception as e:
            logger.error(f"LLM tier routing failed: {e}, using rule-based fallback")
            return self._rule_based_routing(user_message, assistant_response)
    
    def _rule_based_routing(self, user_message: str, assistant_response: str) -> TierRoutingResult:
        """Fallback rule-based routing when LLM is unavailable."""
        user_lower = user_message.lower()
        recommended_tiers = [MemoryTier.SHORT_TERM]  # Always include short-term
        reasoning = {"short_term": "Always stored for working context"}
        
        # Rule-based heuristics
        
        # Episodic patterns (personal experiences/events and reactions)
        episodic_patterns = [
            "i went", "i found", "i saw", "i did", "i was", "i watched",
            "yesterday", "today", "last week", "this morning", "when i",
            "happened", "experience", "story", "event", 
            "that movie", "the movie", "it was", "was terrible", "was great",
            "loved it", "hated it", "enjoyed it", "didn't like it"
        ]
        
        # Semantic patterns (facts about entities)
        semantic_patterns = [
            "my name is", "my mom", "my dad", "my dog", "my cat",
            "lives in", "works at", "is a", "age", "born"
        ]
        
        # Procedural patterns (preferences/rules)
        procedural_patterns = [
            "i like", "i don't like", "i prefer", "i hate", "i love",
            "remind me", "always", "never", "favorite"
        ]
        
        if any(pattern in user_lower for pattern in episodic_patterns):
            recommended_tiers.append(MemoryTier.EPISODIC)
            reasoning["episodic"] = "Rule-based: Contains experience/event patterns"
        
        if any(pattern in user_lower for pattern in semantic_patterns):
            recommended_tiers.append(MemoryTier.SEMANTIC)
            reasoning["semantic"] = "Rule-based: Contains factual/entity patterns"
        
        if any(pattern in user_lower for pattern in procedural_patterns):
            recommended_tiers.append(MemoryTier.PROCEDURAL)
            reasoning["procedural"] = "Rule-based: Contains preference/rule patterns"
        
        # If no specific patterns found, default to episodic for substantial messages
        if len(recommended_tiers) == 1 and len(user_message.strip()) > 10:
            recommended_tiers.append(MemoryTier.EPISODIC)
            reasoning["episodic"] = "Rule-based: Default for substantial messages"
        
        return TierRoutingResult(
            recommended_tiers=recommended_tiers,
            reasoning=reasoning
        )


# Factory function
async def create_memory_tier_router() -> MemoryTierRouter:
    """Create memory tier router."""
    return MemoryTierRouter()